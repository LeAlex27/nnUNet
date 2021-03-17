import torch
from torch import nn
from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.utilities.nd_softmax import softmax_helper


class CountingUNet(Generic_UNet):
    def __init__(self, **unet_kw):
        unet_kw['final_nonlin'] = lambda x: x
        unet_kw['num_classes'] = 3
        super(CountingUNet, self).__init__(**unet_kw)

        conv_kwargs = {'kernel_size': 3,
                       'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

    def forward(self, x):
        return super(CountingUNet, self).forward(x)
