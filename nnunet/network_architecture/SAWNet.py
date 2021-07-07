import numpy as np
import torch
from torch import nn
from batchgenerators.augmentations.utils import pad_nd_image
from nnunet.network_architecture.generic_UNet import Generic_UNet, StackedConvLayers, Upsample
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.generic_UNet import ConvDropoutNormNonlin
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch
from typing import Union, Tuple


class SAUnit(nn.Module):
    def __init__(self, n_channels):
        super(SAUnit, self).__init__()
        # print("SAWNet.py:13 n_channels:", n_channels)

        conv_kw = {'in_channels': n_channels,
                   'out_channels': n_channels,
                   'kernel_size': (1, 1)}

        self.f_conv = nn.Conv2d(**conv_kw)
        self.g_conv = nn.Conv2d(**conv_kw)
        self.h_conv = nn.Conv2d(**conv_kw)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout2d(**{'p': 0.5})  #, 'inplace': True})
        self.conv = nn.Conv2d(**conv_kw)
        self.nonlin = nn.LeakyReLU(**{'negative_slope': 1e-2})  #, 'inplace': True})
        self.bn = nn.BatchNorm2d(n_channels, **{'eps': 1e-5, 'affine': True, 'momentum': 0.9})

    def forward(self, x):
        # print("SAWNet.py:29", x.size())

        new_shape = (list(x.size())[0], list(x.size())[1], -1)
        # print("new_shape", new_shape)
        f = self.f_conv(x).reshape(new_shape)
        # print("f.shape:", f.shape)
        g = torch.transpose(self.g_conv(x).reshape(new_shape), 1, 2)
        # print("g.shape:", g.shape)
        h = self.h_conv(x).reshape(new_shape)
        # print("h.shape:", h.shape)

        fg = torch.matmul(f, g)
        fg = self.dropout(self.softmax(fg))
        # print("fg.size:", fg.size())
        hfg = torch.matmul(fg, h).reshape(x.size())

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
        self.w_outputs = []

        upsample_mode = 'bilinear'
        transpconv = nn.ConvTranspose2d

        output_features = base_num_features
        input_features = input_channels
        print("num_pool:", num_pool)
        print("input_features:", input_features)
        for d in range(num_pool):
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)
            print("{}: {}".format(d, input_features))

        self.sau = SAUnit(output_features)

        if self.convolutional_upsampling:
            final_num_features = output_features  # base_num_features
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
                # print("u: {} nfeatures_from_down: {} nfeatures_from_skip: {}".format(u, nfeatures_from_down,
                #                                                                     nfeatures_from_skip))
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

        for ds in range(len(self.conv_blocks_w)):
            # todo: relu?
            self.w_outputs.append(conv_op(self.conv_blocks_w[ds][-1].output_channels, 1,
                                          1, 1, 0, 1, 1, seg_output_use_bias))

        self.conv_blocks_w = nn.ModuleList(self.conv_blocks_w)
        self.tuw = nn.ModuleList(self.tuw)
        self.w_outputs = nn.ModuleList(self.w_outputs)
        # self.final_conv = nn.Conv2d(32, 1, (1, 1))

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x):
        # if torch.sum(torch.isinf(x)) > 0 or torch.sum(torch.isnan(x)) > 0:
        #    print("SAWNet.py:137: encountered inf/nan:")
        #    print("x:", x.size(), torch.sum(torch.isinf(x)), torch.sum(torch.isnan(x)))

        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            # if torch.sum(torch.isinf(x)) > 0 or torch.sum(torch.isnan(x)) > 0:
            #    print("SAWNet.py:145: encountered inf/nan in block d: {}:".format(d))
            #    print("x:", x.size(), torch.sum(torch.isinf(x)), torch.sum(torch.isnan(x)))
            #    exit(1)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)
        sau_x = self.sau(x.clone())

        for u in range(len(self.tu)):
            # print("{} x.shape: {}".format(u, x.shape))
            x = self.tu[u](x)
            # print(x.shape, skips[-(u + 1)].shape)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            # if torch.sum(torch.isinf(x)) > 0 or torch.sum(torch.isnan(x)) > 0:
            #    print("SAWNet.py:162: encountered inf/nan:")
            #    print("x:", x.size(), torch.sum(torch.isinf(x)), torch.sum(torch.isnan(x)))
            # print("SN:144 {}".format(x.shape))
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        saw_outputs = []
        for u in range(len(self.tuw)):
            # print("{} sau_x.shape".format(u), sau_x.shape)
            sau_x = self.tuw[u](sau_x)
            # print("skips.shape:", skips[-(u + 1)].shape, sau_x.shape)
            sau_x = torch.cat((sau_x, skips[-(u + 1)]), dim=1)
            sau_x = self.conv_blocks_w[u](sau_x)
            saw_outputs.append(self.w_outputs[u](sau_x))

        if self._deep_supervision and self.do_ds:
            assert self.upscale_logits is False
            return tuple([torch.cat((seg_outputs[-1], saw_outputs[-1]), dim=1)]
                         + [torch.cat((i(j), i(k)), dim=1) for i, j, k in zip(list(self.upscale_logits_ops)[::-1],
                                                                              seg_outputs[:-1][::-1],
                                                                              saw_outputs[:-1][::-1])])
        else:
            return torch.cat((seg_outputs[-1], saw_outputs[-1]), dim=1)

    @staticmethod
    def compute_approx_vram_consumption(**kw_args):
        return 1.5 * Generic_UNet.compute_approx_vram_consumption(**kw_args)

    def _internal_predict_3D_3Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"
        assert self.get_device() != "cpu"
        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map).cuda(self.get_device(),
                                                                                     non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(data.shape[1:], device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes + 1] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes + 1] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(data.shape[1:], dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes + 1] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes + 1] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring,
                        gaussian_importance_map)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities

    def _internal_maybe_mirror_and_pred_2D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        # everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        # we now return a cuda tensor! Not numpy array!
        assert len(x.shape) == 4, 'x must be (b, c, x, y)'

        x = to_cuda(maybe_to_torch(x), gpu_id=self.get_device())
        result_torch = torch.zeros([x.shape[0], self.num_classes + 1] + list(x.shape[2:]),
                                   dtype=torch.float).cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = to_cuda(maybe_to_torch(mult), gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                sx = self(x)
                pred = self.inference_apply_nonlin(sx)
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_predict_2D_2Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 3, "x must be (c, x, y)"
        assert self.get_device() != "cpu"
        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_2d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_2d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_2d = gaussian_importance_map
                self._patch_size_for_gaussian_2d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_2d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map).cuda(self.get_device(),
                                                                                     non_blocking=True)
        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(data.shape[1:], device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes + 1] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes + 1] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(data.shape[1:], dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes + 1] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes + 1] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]

                predicted_patch = self._internal_maybe_mirror_and_pred_2D(
                    data[None, :, lb_x:ub_x, lb_y:ub_y], mirror_axes, do_mirroring,
                    gaussian_importance_map)[0]

                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities
