import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.morphology import label
from skimage.measure import regionprops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss
from nnunet.utilities.nd_softmax import softmax_helper


class SAWLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(SAWLoss, self).__init__()
        self.dice = SoftDiceLoss(softmax_helper, **{'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
        self.l2 = MSELoss()

    def forward(self, x, y, loss_mask=None):
        print("saw_loss.py:22")

        # create gt density map
        y_cpu = y.cpu().numpy()
        dm = np.empty_like(y_cpu[:, 0:1])
        for i in range(y.shape[0]):
            dm[i, 0] = self.sharpen(y_cpu[i, 0])
            # dm[i, 0] = self.pixel(y_cpu[i, 0])
        # self.save_img(dm, '/cluster/husvogt/debug_imgs/{:04d}_{:03d}.png')
        dm = torch.from_numpy(dm).cuda()
        y_n_ma = torch.sum(dm)
        x_n_ma = torch.sum(x[:, 3]) # -1: = 3:

        print("sum x:", x_n_ma)
        print("sum dm:", y_n_ma)

        l_ = self.dice(x[:, :2], y) #, loss_mask=loss_mask)
        print("l_:", l_)
        # print("shapes", x.shape, dm.shape)
        l_dm = self.loss_density_map(softmax_helper(x[:, 2:]), dm)
        print("l_dm:", l_dm)
        l_n = self.loss_n_ma(x_n_ma, y_n_ma)
        print("l_n:", l_n)

        return l_ + l_dm + 1e-6 * l_n  # + l_dm + l_n

    def save_img(self, img, fname):
        fig, ax = plt.subplots(1, img.shape[0], figsize=(10 * img.shape[0],  10))
        for i in range(img.shape[0]):
            ax[i].imshow(img[i, 0])
        fig.savefig(fname.format(self.n, img.shape[2]))
        self.n += 1

    @staticmethod
    def labels_and_props(img):
        labels = skimage.morphology.label(img)
        props = skimage.measure.regionprops(labels)

        return labels, props

    @staticmethod
    def sharpen(img):
        t = np.zeros_like(img)

        labels, props = SAWLoss.labels_and_props(img)

        srpi = np.sqrt(2 * np.pi)
        s = 2.0
        k = 10e10
        s_k = s ** (1.0 / k)

        for p_ in props:
            for i in range(p_.bbox[0], p_.bbox[2]):
                for j in range(p_.bbox[1], p_.bbox[3]):
                    sum_ = 0
                    for p in props:
                        p_i = int(p.centroid[0])
                        p_j = int(p.centroid[1])
                        sum_ += 1.0 / (srpi * s_k) * np.exp(-((i - p_i) ** 2 + (j - p_j) ** 2) / (2.0 * s_k ** 2.0))
                    t[i, j] = sum_

        return t / 2.50635

    @staticmethod
    def pixel(img):
        t = np.zeros_like(img)
        _, props = SAWLoss.labels_and_props(img)

        for p in props:
            p_i = int(p.centroid[0])
            p_j = int(p.centroid[1])
            t[p_i, p_j] = 1

        return t

