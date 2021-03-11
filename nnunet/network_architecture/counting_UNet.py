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

        # self.conv_0 = nn.Conv2d(1, 4, **conv_kwargs)
        # self.bn_0 = nn.BatchNorm2d(4, **norm_op_kwargs)
        # self.relu_0 = nn.LeakyReLU(**nonlin_kwargs)
        # self.conv_1 = nn.Conv2d(4, 8, **conv_kwargs)
        # self.bn_1 = nn.BatchNorm2d(8, **norm_op_kwargs)
        # self.relu_1 = nn.LeakyReLU(**nonlin_kwargs)
        # self.pool_op_0 = nn.MaxPool2d(2)

        # self.conv_2 = nn.Conv2d(8, 16, **conv_kwargs)
        # self.bn_2 = nn.BatchNorm2d(16, **norm_op_kwargs)
        # self.relu_2 = nn.LeakyReLU(**nonlin_kwargs)
        # self.conv_3 = nn.Conv2d(16, 32, **conv_kwargs)
        # self.bn_3 = nn.BatchNorm2d(32, **norm_op_kwargs)
        # self.relu_3 = nn.LeakyReLU(**nonlin_kwargs)
        # self.pool_op_1 = nn.MaxPool2d(2)

    def forward(self, x):
        x = super(CountingUNet, self).forward(x)

        # x = self.conv_0(x_0[:, 1:])
        # x = self.relu_0(self.bn_0(x))
        # x = self.conv_1(x)
        # x = self.relu_1(self.bn_1(x))
        # x = self.pool_op_0(x)

        # x = self.conv_2(x)
        # x = self.relu_2(self.bn_2(x))
        # x = self.conv_3(x)
        # x = self.relu_3(self.bn_3(x))
        # x_1 = self.pool_op_1(x)
        #print("x_0.shape:", x_0.shape)
        #print("x_1.shape:", x_1.shape)

        #print("x.shape:", x.shape)
        print("counting_UNet.py:52")
        print(len(x))
        for i in x:
            print(i.shape)
        #x = torch.cat((softmax_helper(x[:, :2])[:, :1],
        #              x[:, 2:]), 1)

        return x
