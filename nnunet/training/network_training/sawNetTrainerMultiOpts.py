import torch
from torch import nn
from time import time
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.SAWNet import SAWNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast, GradScaler
from nnunet.training.loss_functions.counting_dice_loss import CountingDiceLoss
from nnunet.training.learning_rate.poly_lr import poly_lr
import numpy as np
import pickle
from collections import OrderedDict
from torch.optim.lr_scheduler import _LRScheduler
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.dataloading.dataset_loading import load_dataset
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from batchgenerators.dataloading import MultiThreadedAugmenter
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss

from batchgenerators.transforms import DataChannelSelectionTransform, SegChannelSelectionTransform, SpatialTransform, \
    GammaTransform, MirrorTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params
from nnunet.training.data_augmentation.downsampling import DownsampleSegForDSTransform3, DownsampleSegForDSTransform2
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgenerators.transforms.abstract_transforms import AbstractTransform


class DensityMapTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        y = data_dict['seg']
        print("sawNetTrainerMultiOpts.py:46")
        print(y.shape)
        new_y = np.empty((y.shape[0], y.shape[1] + 1, y.shape[2], y.shape[3]), dtype=y.dtype)
        new_y[:, :1] = y
        for i in range(y.shape[0]):
            new_y[i, 1] = CountingDiceLoss.sharpen(y[i, 0])
        data_dict['seg'] = new_y
        return data_dict


class sawNetTrainerMultiOpts(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super(sawNetTrainerMultiOpts, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice,
                                                     stage, unpack_data, deterministic, fp16, False)
        self.max_num_epochs = 2
        self.loss = None
        self.opt_loss = []

        self.initial_lr = None
        self.initial_lrs = [5e-4, 1e-4]
        self.pickle_losses = []

        print("sawNetTrainerTwoOpts:")
        print("output folder:", self.output_folder)
        print("epochs:", self.max_num_epochs)

    def __del__(self):
        if self.output_folder is not None:
            with open(self.output_folder + '/losses.pickle', 'wb') as f:
                for i in range(len(self.pickle_losses)):
                    pickle.dump(self.pickle_losses[i], f)

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        print("sawNetTrainerMultiOpts.py:89 initialize")
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            # self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights)
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = self.get_moreDA_augmentation_depth_map(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            breakpoint()
            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        print("sawNetTrainerMultiOpts.py:155 inititalize network")
        assert self.threeD is False
        conv_op = nn.Conv2d
        dropout_op = nn.Dropout2d
        norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # print("sawNetTrainerMultiOpts.py:62", self.net_num_pool_op_kernel_sizes)
        self.network = SAWNet(self.num_input_channels, self.base_num_features, self.num_classes,
                              len(self.net_num_pool_op_kernel_sizes),
                              self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                              dropout_op_kwargs,            # ds below
                              net_nonlin, net_nonlin_kwargs, False, False, softmax_helper, InitWeights_He(1e-2),
                              self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        print("sawNetTrainerMultiOpts.py:176 inititalize optimizer and scheduler")
        self.opt_loss.append((torch.optim.Adam(self.network.parameters(), self.initial_lrs[0]),
                              # CountingDiceLoss(True, False, False, None)))
                              SoftDiceLoss(softmax_helper, **{'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})))
        self.opt_loss.append((torch.optim.Adam(self.network.parameters(), self.initial_lrs[1]),
                              # CountingDiceLoss(False, True, True, None)))
                              torch.nn.MSELoss()))
        # self.opt_loss.append((torch.optim.Adam(self.network.parameters(), self.initial_lrs[2]),
        #                       CountingDiceLoss(False, False, True, None)))
        self.pickle_losses = [[] for _ in range(len(self.opt_loss))]

    def maybe_update_lr(self, epoch=None):
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch

        for idx, (opt, _) in enumerate(self.opt_loss):
            opt.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lrs[idx], 0.9)
            self.print_to_log_file("lr:", np.round(opt.param_groups[0]['lr'], decimals=6))

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler,
                                                     'state_dict'):  # not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            # WTF is this!?
            # for key in lr_sched_state_dct.keys():
            #    lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = [opt.state_dict() for opt, _ in self.opt_loss]
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        save_this = {
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics),
            'best_stuff' : (self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA)}
        if self.amp_grad_scaler is not None:
            save_this['amp_grad_scaler'] = self.amp_grad_scaler.state_dict()

        torch.save(save_this, fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time() - start_time))

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        print("sawNetTrainerMultiOpts.py:230 run iteration")
        # with torch.autograd.set_detect_anomaly(True):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        if not self.fp16:
            print("supports only mixed precision")
            exit(1)

        for idx, (opt, loss) in enumerate(self.opt_loss):
            opt.zero_grad()

            with autocast():
                output = [self.network(data)]
                print("sawNewTrainerMultiOpts.py:257")
                print(output.size())
                print(target.size())
                if idx == 0:
                    l = loss(output[0], target[0])
                elif idx == 1:
                    l = loss(output[0], target[0])

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(opt)
                self.amp_grad_scaler.update()

            self.pickle_losses[idx].append(l.detach().cpu().numpy())
        del data

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(self.network.state_dict().keys())
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                for idx, (opt, _) in enumerate(self.opt_loss):
                    if optimizer_state_dict[idx] is not None:
                        opt.load_state_dict(optimizer_state_dict[idx])

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            #self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            #self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                #self.print_to_log_file("saving best epoch checkpoint...")
                if self.save_best_checkpoint: self.save_checkpoint(join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                #self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                pass
                #self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                all_lr_below_threshold = True
                for opt, _ in self.opt_loss:
                    if opt.param_groups[0]['lr'] > self.lr_threshold:
                        all_lr_below_threshold = False

                if all_lr_below_threshold:
                    continue_training = False
            else:
                pass
                #self.print_to_log_file(
                #    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    @staticmethod
    def get_moreDA_augmentation_depth_map(dataloader_train, dataloader_val, patch_size,
                                          params=default_3D_augmentation_params, border_val_seg=-1,
                                          seeds_train=None, seeds_val=None, order_seg=1, order_data=3,
                                          deep_supervision_scales=None, soft_ds=False,
                                          classes=None, pin_memory=True, regions=None,
                                          use_nondetMultiThreadedAugmenter: bool = False):
        assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"
        assert use_nondetMultiThreadedAugmenter is False

        tr_transforms = [DensityMapTransform()]

        if params.get("selected_data_channels") is not None:
            tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

        if params.get("selected_seg_channels") is not None:
            tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if params.get("dummy_2D") is not None and params.get("dummy_2D"):
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
        else:
            ignore_axes = None

        tr_transforms.append(SpatialTransform(
            patch_size, patch_center_dist_from_border=None,
            do_elastic_deform=params.get("do_elastic"), alpha=params.get("elastic_deform_alpha"),
            sigma=params.get("elastic_deform_sigma"),
            do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
            angle_z=params.get("rotation_z"), p_rot_per_axis=params.get("rotation_p_per_axis"),
            do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
            border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=order_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg,
            order_seg=order_seg, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
            p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
            independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
        ))

        if params.get("dummy_2D"):
            tr_transforms.append(Convert2DTo3DTransform())

        # we need to put the color augmentations after the dummy 2d part (if applicable). Otherwise the overloaded color
        # channel gets in the way
        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

        if params.get("do_additive_brightness"):
            tr_transforms.append(BrightnessTransform(params.get("additive_brightness_mu"),
                                                     params.get("additive_brightness_sigma"),
                                                     True, p_per_sample=params.get("additive_brightness_p_per_sample"),
                                                     p_per_channel=params.get("additive_brightness_p_per_channel")))

        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=ignore_axes))
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), True, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=0.1))  # inverted gamma

        if params.get("do_gamma"):
            tr_transforms.append(
                GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                               p_per_sample=params["p_gamma"]))

        if params.get("do_mirror") or params.get("mirror"):
            tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

        if params.get("mask_was_used_for_normalization") is not None:
            mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
            tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
            tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
            if params.get("cascade_do_cascade_augmentations") is not None and params.get(
                    "cascade_do_cascade_augmentations"):
                if params.get("cascade_random_binary_transform_p") > 0:
                    tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                        p_per_sample=params.get("cascade_random_binary_transform_p"),
                        key="data",
                        strel_size=params.get("cascade_random_binary_transform_size"),
                        p_per_label=params.get("cascade_random_binary_transform_p_per_label")))
                if params.get("cascade_remove_conn_comp_p") > 0:
                    tr_transforms.append(
                        RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                            channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                            key="data",
                            p_per_sample=params.get("cascade_remove_conn_comp_p"),
                            fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                            dont_do_if_covers_more_than_X_percent=params.get(
                                "cascade_remove_conn_comp_fill_with_other_class_p")))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                tr_transforms.append(DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
            else:
                tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
                                                                  output_key='target'))

        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)

        batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
                                                      params.get("num_cached_per_thread"),
                                                      seeds=seeds_train, pin_memory=pin_memory)

        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))
        if params.get("selected_data_channels") is not None:
            val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
        if params.get("selected_seg_channels") is not None:
            val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

        if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
            val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

        val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

        if deep_supervision_scales is not None:
            if soft_ds:
                assert classes is not None
                val_transforms.append(
                    DownsampleSegForDSTransform3(deep_supervision_scales, 'target', 'target', classes))
            else:
                val_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, 0, input_key='target',
                                                                   output_key='target'))

        val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        val_transforms = Compose(val_transforms)

        batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms,
                                                    max(params.get('num_threads') // 2, 1),
                                                    params.get("num_cached_per_thread"),
                                                    seeds=seeds_val, pin_memory=pin_memory)

        return batchgenerator_train, batchgenerator_val
