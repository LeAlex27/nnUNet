import torch
from torch import nn
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.SAWNet import SAWNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.cuda.amp import autocast
from nnunet.training.loss_functions.counting_dice_loss import CountingDiceLoss
import pickle


class sawNetTrainerMultiOpts(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super(sawNetTrainerMultiOpts, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice,
                                                     stage, unpack_data, deterministic, fp16, False)
        self.max_num_epochs = 300
        self.loss = None
        self.opt_loss = []

        self.pickle_losses = []

        print("sawNetTrainerTwoOpts:")
        print("output folder:", self.output_folder)
        print("epochs:", self.max_num_epochs)

    def __del__(self):
        if self.output_folder is not None:
            with open(self.output_folder + '/losses.pickle', 'wb') as f:
                pickle.dump(self.l_, f)
                pickle.dump(self.l_dm, f)
                # pickle.dump(self.l_n, f)

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        assert self.threeD is False
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            norm_op = nn.InstanceNorm3d

        else:
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
        self.opt_loss.append((torch.optim.Adam(self.network.parameters(), 1e-3),
                              CountingDiceLoss(True, False, False, None)))
        self.opt_loss.append((torch.optim.Adam(self.network.parameters(), 1e-4),
                              CountingDiceLoss(False, True, False, None)))
        self.pickle_losses = [[], []]

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
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
                output = self.network(data)
                del data
                l = loss(output, target[0])

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(opt)
                self.amp_grad_scaler.update()

            self.pickle_losses[idx].append(l.detach().cpu().numpy())

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()
