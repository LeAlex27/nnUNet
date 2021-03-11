import torch
from torch import nn
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.network_architecture.counting_UNet import CountingUNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.counting_dice_loss import CountingDiceLoss


class CountingTrainer(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = CountingDiceLoss()
        self.max_num_epochs = 2

    def initialize_network(self):
        """
        This is specific to the U-Net and must be adapted for other network architectures
        :return:
        """
        net_numpool = len(self.net_num_pool_op_kernel_sizes)

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
        unet_kw = {'input_channels': self.num_input_channels, 'base_num_features': self.base_num_features,
                   'num_classes': self.num_classes, 'num_pool': net_numpool, 'num_conv_per_stage': self.conv_per_stage,
                   'feat_map_mul_on_downscale': 2, 'conv_op': conv_op, 'norm_op': norm_op,
                   'norm_op_kwargs': norm_op_kwargs, 'dropout_op': dropout_op, 'dropout_op_kwargs': dropout_op_kwargs,
                   'nonlin': net_nonlin, 'nonlin_kwargs': net_nonlin_kwargs, 'deep_supervision': True,
                   'dropout_in_localization': False, 'final_nonlin': lambda x: x,
                   'weightInitializer': InitWeights_He(1e-2), 'pool_op_kernel_sizes': self.net_num_pool_op_kernel_sizes,
                   'conv_kernel_sizes': self.net_conv_kernel_sizes, 'upscale_logits': False,
                   'convolutional_pooling': True, 'convolutional_upsampling': True, 'max_num_features': None,
                   'seg_output_use_bias': False}
        self.network = CountingUNet(**unet_kw)
        self.network.inference_apply_nonlin = softmax_helper

        if torch.cuda.is_available():
            self.network.cuda()