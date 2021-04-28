import numpy as np
import torch
from torch import nn
from nnunet.network_architecture.generic_UNet import Generic_UNet, StackedConvLayers, Upsample
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin


class SAUnit(nn.Module):
    def __init__(self, n_channels):
        super(SAUnit, self).__init__()\

        conv_kw = {'in_channels': n_channels,
                   'out_channels': n_channels,
                   'kernel_size': (1, 1)}

        self.f_conv = nn.Conv2d(**conv_kw)
        self.g_conv = nn.Conv2d(**conv_kw)
        self.h_conv = nn.Conv2d(**conv_kw)

        self.dropout = nn.Dropout2d(**{'p': 0.5, 'inplace': True})
        self.conv = nn.Conv2d(**conv_kw)
        self.nonlin = nn.LeakyReLU(**{'negative_slope': 1e-2, 'inplace': True})
        self.bn = nn.BatchNorm2d(**{'eps': 1e-5, 'affine': True, 'momentum': 0.1})

    def forward(self, x):
        print("SAWNet.py:25", x.size)

        new_shape = (x.size[0], x.size[1], -1)
        print("new_shape", new_shape)
        f = self.f_conv(x).reshape(new_shape)
        g = torch.transpose(self.g_conv(x).reshape(new_shape), 1, 2)
        h = self.h_conv(x).reshape(new_shape)

        fg = self.dropout(nn.functional.softmax(torch.matmul(f, g), dim=1))
        hfg = torch.matmul(h, fg).reshape(x.size)

        hfg = self.bn(self.nonlin(self.conv(hfg)))
        return hfg + x


class SAWNet(Generic_UNet):
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
        super(SAWNet, self).__init__(input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage,
                 feat_map_mul_on_downscale, conv_op, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                 nonlin, nonlin_kwargs, deep_supervision, dropout_in_localization, final_nonlin, weightInitializer,
                                     pool_op_kernel_sizes, conv_kernel_sizes, upscale_logits, convolutional_pooling,
                                     convolutional_upsampling, max_num_features, basic_block, seg_output_use_bias)

        self.conv_blocks_w = []
        self.tuw = []

        upsample_mode = 'bilinear'
        transpconv = nn.ConvTranspose2d

        for d in range(num_pool):
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

        self.sau = SAUnit(input_features)

        if self.convolutional_upsampling:
            final_num_features = base_num_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tuw.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tuw.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                           pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_blocks_w.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        self.conv_blocks_w = nn.ModuleList(self.blocks_w)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x):
        print("SAWNet.py:ca66", x.size)
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        sau_x = self.sau(x)
        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        saw_outputs = []
        for u in range(len(self.tuw)):
            sau_x = self.tuw[u](sau_x)
            sau_x = torch.cat((sau_x, skips[-(u + 1)]), dim=1)
            sau_x = self.conv_blocks_w[u](sau_x)
            saw_outputs.append(sau_x)

        if self._deep_supervision and self.do_ds:
            assert self.upscale_logits is False
            return tuple([torch.cat((seg_outputs[-1], saw_outputs[-1]), dim=1)]
                         + [torch.cat((i(j), i(k)), dim=1) for i, j, k in zip(list(self.upscale_logits_ops)[::-1],
                                                                              seg_outputs[:-1][::-1],
                                                                              saw_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1], saw_outputs[-1]  # todo: cat

    @staticmethod
    def compute_approx_vram_consumption(**kw_args):
        return 1.5 * Generic_UNet.compute_approx_vram_consumption(**kw_args)