import torch
from torch import nn
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.SAWNet import SAWNet
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.training.loss_functions.counting_dice_loss import CountingDiceLoss
from nnunet.utilities.nd_softmax import softmax_helper
import numpy as np


class sawNetTrainer(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super(sawNetTrainer, self).__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
                                            unpack_data, deterministic, fp16, False)
        self.loss = CountingDiceLoss(label_loss=True, density_map_loss=True, count_loss=True,
                                     output_folder=self.output_folder)
        self.optimizer = 'adam'
        self.max_num_epochs = 200
        self.initial_lr = 1e-3
        self.use_lr_scheduler = False

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
        print("sawNetTrainer.py:46", self.net_num_pool_op_kernel_sizes)
        self.network = SAWNet(self.num_input_channels, self.base_num_features, self.num_classes,
                              len(self.net_num_pool_op_kernel_sizes),
                              self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                              dropout_op_kwargs,            # ds below
                              net_nonlin, net_nonlin_kwargs, False, False, softmax_helper, InitWeights_He(1e-2),
                              self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)
        if torch.cuda.is_available():
            self.network.cuda()

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        if self.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                             momentum=0.99, nesterov=True)
        elif self.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr)

            if self.use_lr_scheduler:
                def cosine_wwr(step):
                    t_mul = 2.0
                    m_mul = 1.0
                    alpha = 0.0
                    first_decay_steps = 50

                    global_step_recomp = step
                    completed_fraction = global_step_recomp / first_decay_steps

                    i_restart = np.floor(np.log(1.0 - completed_fraction * (1.0 - t_mul)) / np.log(t_mul))
                    sum_r = (1.0 - t_mul ** i_restart) / (1.0 - t_mul)
                    completed_fraction = (completed_fraction - sum_r) / t_mul ** i_restart

                    m_fac = m_mul ** i_restart
                    cosine_decayed = 0.5 * m_fac * (1.0 + np.cos(np.pi * completed_fraction))
                    decayed = (1 - alpha) * cosine_decayed + alpha

                    return decayed

                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, cosine_wwr)
