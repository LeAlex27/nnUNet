import numpy as np
from nnunet.training.dataloading.dataset_loading import DataLoader2D
from nnunet.training.loss_functions.counting_dice_loss import CountingDiceLoss
from batchgenerators.utilities.file_and_folder_operations import *


class DMLoader(DataLoader2D):
    def __init__(self, data, patch_size, final_patch_size, batch_size, oversample_foreground_percent=0.0,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        super(DMLoader, self).__init__(data, patch_size, final_patch_size, batch_size, oversample_foreground_percent,
                                       memmap_mode, pseudo_3d_slices, pad_mode, pad_kwargs_data, pad_sides)
        self._loaded = dict()

        for i in self._data.keys():
            d = np.load(self._data[i]['data_file'][:-4] + ".npy")
            new_shape = (d.shape[0] + 1, d.shape[1], d.shape[2], d.shape[3])
            d_dm = np.empty(new_shape, d.dtype)
            d_dm[:2] = d
            d_t = np.where(d_dm[1, 0] < 0, 0, d_dm[1, 0])
            d_dm[2, 0] = CountingDiceLoss.sharpen(d_t)
            self._loaded[i] = d_dm

    def determine_shapes(self):
        num_seg = 2

        k = list(self._data.keys())[0]
        if isfile(self._data[k]['data_file'][:-4] + ".npy"):
            case_all_data = np.load(self._data[k]['data_file'][:-4] + ".npy", self.memmap_mode)
        else:
            case_all_data = np.load(self._data[k]['data_file'])['data']
        num_color_channels = case_all_data.shape[0] - 1 # num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        selected_keys = np.random.choice(self.list_of_keys, self.batch_size, True, None)

        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        case_properties = []
        for j, i in enumerate(selected_keys):
            if 'properties' in self._data[i].keys():
                properties = self._data[i]['properties']
            else:
                properties = load_pickle(self._data[i]['properties_file'])
            case_properties.append(properties)

            if self.get_do_oversample(j):
                force_fg = True
            else:
                force_fg = False

            # case_all_data = np.load(self._data[i]['data_file'][:-4] + ".npy", self.memmap_mode)
            case_all_data = self._loaded[i]

            # this is for when there is just a 2d slice in case_all_data (2d support)
            if len(case_all_data.shape) == 3:
                case_all_data = case_all_data[:, None]

            # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
            if not force_fg:
                random_slice = np.random.choice(case_all_data.shape[1])
                selected_class = None
            else:
                # these values should have been precomputed
                if 'class_locations' not in properties.keys():
                    raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

                foreground_classes = np.array(
                    [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
                foreground_classes = foreground_classes[foreground_classes > 0]
                if len(foreground_classes) == 0:
                    selected_class = None
                    random_slice = np.random.choice(case_all_data.shape[1])
                    print('case does not contain any foreground classes', i)
                else:
                    selected_class = np.random.choice(foreground_classes)

                    voxels_of_that_class = properties['class_locations'][selected_class]
                    valid_slices = np.unique(voxels_of_that_class[:, 0])
                    random_slice = np.random.choice(valid_slices)
                    voxels_of_that_class = voxels_of_that_class[voxels_of_that_class[:, 0] == random_slice]
                    voxels_of_that_class = voxels_of_that_class[:, 1:]

            # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
            # below the current slice, here is where we get them. We stack those as additional color channels
            assert self.pseudo_3d_slices == 1
            if self.pseudo_3d_slices == 1:
                case_all_data = case_all_data[:, random_slice]

            # case all data should now be (c, x, y)
            assert len(case_all_data.shape) == 3

            # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
            # define what the upper and lower bound can be to then sample form them with np.random.randint

            need_to_pad = self.need_to_pad
            for d in range(2):
                # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
                # always
                if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                    need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

            shape = case_all_data.shape[1:]
            lb_x = - need_to_pad[0] // 2
            ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
            lb_y = - need_to_pad[1] // 2
            ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

            # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
            # at least one of the foreground classes in the patch
            if not force_fg or selected_class is None:
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            else:
                # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)

            bbox_x_ub = bbox_x_lb + self.patch_size[0]
            bbox_y_ub = bbox_y_lb + self.patch_size[1]

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_x_lb = max(0, bbox_x_lb)
            valid_bbox_x_ub = min(shape[0], bbox_x_ub)
            valid_bbox_y_lb = max(0, bbox_y_lb)
            valid_bbox_y_ub = min(shape[1], bbox_y_ub)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)

            case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub, valid_bbox_y_lb:valid_bbox_y_ub]

            # assert case_all_data[:2] == 3
            case_all_data_donly = np.pad(case_all_data[:1], ((0, 0),
                                                             (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                             (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                         self.pad_mode, **self.pad_kwargs_data)

            case_all_data_segonly = np.pad(case_all_data[1:], ((0, 0),
                                                               (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                               (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                           'constant', **{'constant_values': -1})

            data[j] = case_all_data_donly
            seg[j] = case_all_data_segonly

        keys = selected_keys
        return {'data': data, 'seg': seg, 'properties': case_properties, "keys": keys}