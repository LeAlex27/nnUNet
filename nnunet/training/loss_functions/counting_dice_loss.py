import numpy as np
import skimage
import torch
from skimage.morphology import label
from skimage.measure import regionprops
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss
from nnunet.training.loss_functions.focal_loss import FocalLoss
from nnunet.utilities.nd_softmax import softmax_helper


class CountingDiceLoss(torch.nn.Module):
    def __init__(self, alpha=0.01):
        super(CountingDiceLoss, self).__init__()
        self.loss = SoftDiceLoss(softmax_helper, **{'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
        self.loss_density_map = torch.nn.MSELoss()
        self.loss_n_ma = torch.nn.MSELoss()

    def forward(self, x, y, loss_mask=None):
        print("counting_dice_loss.py:22")

        # create gt density map
        y_cpu = y.cpu().numpy()
        dm = np.empty_like(y_cpu[:, 0:1])
        for i in range(y.shape[0]):
            dm[i, 0] = self.sharpen(y_cpu[i, 0])
        dm = torch.from_numpy(dm).cuda()
        y_n_ma = torch.sum(dm)
        x_n_ma = torch.sum(x[:, -1:])

        print("sum x:", torch.sum(x[:, -1:]))
        print("sum dm:", y_n_ma)

        l_ = self.loss(x[:, :-1], y) #, loss_mask=loss_mask)
        print("l_:", l_)
        # print("shapes", x.shape, dm.shape)
        l_dm = self.loss_density_map(x[:, -1:], dm)
        print("l_dm:", l_dm)
        l_n = self.loss_n_ma(x_n_ma, y_n_ma)
        print("l_n:", l_n)

        return l_dm # l_ + l_dm + 0.001 * l_n

    @staticmethod
    def labels_and_props(img):
        labels = skimage.morphology.label(img)
        props = skimage.measure.regionprops(labels)

        return labels, props

    @staticmethod
    def sharpen(img):
        t = np.zeros_like(img)

        labels, props = CountingDiceLoss.labels_and_props(img)

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
