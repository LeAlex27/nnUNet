import torch
from torch import nn
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.SAWNet import SAWNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.counting_dice_loss import CountingDiceLoss


class sawNetTrainer(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super(sawNetTrainer, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.loss = CountingDiceLoss()
        self.max_num_epochs = 10

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
        self.network = SAWNet(self.num_input_channels, self.base_num_features, self.num_classes,
                              len(self.net_num_pool_op_kernel_sizes),
                              self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                              dropout_op_kwargs,
                              net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                              self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper
