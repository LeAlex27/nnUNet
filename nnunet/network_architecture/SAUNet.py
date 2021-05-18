from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.SAWNet import SAUnit
from nnunet.network_architecture.generic_UNet import Generic_UNet, ConvDropoutNormNonlin
import torch.nn.functional


class SAUNet(Generic_UNet):
    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        super(SAUNet, self).__init__(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                 feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs,
                 dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization,
                 final_nonlin, weightInitializer, pool_op_kernel_sizes,
                 conv_kernel_sizes, upscale_logits, convolutional_pooling, convolutional_upsampling,
                 max_num_features, basic_block, seg_output_use_bias)

        output_features = base_num_features
        for d in range(num_pool):
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)
            print("{}: {}".format(d, input_features))

        self.sau = SAUnit(output_features)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)
        x = self.sau(x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
