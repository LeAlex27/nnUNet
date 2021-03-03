import numpy as np
import skimage
import torch
from skimage.morphology import label
from skimage.measure import regionprops
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss


def labels_and_props(img):
    labels = skimage.morphology.label(img)
    props = skimage.measure.regionprops(labels)

    return labels, props

def sharpen(img):
    t = np.zeros_like(img)

    labels, props = labels_and_props(img)

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


class CountingDiceLoss(SoftDiceLoss):
    def __init__(self, **sd_kw):
        super(CountingDiceLoss, self).__init__(**sd_kw)

    def forward(self, x, y, loss_mask=None):
        print("cdLoss:")
        print("\tx.shape:", x.shape)
        print("\ty.shape:", y.shape)

        dm = np.empty(x.shape)
        idxs = y.shape[0]
        for i in range(idxs):
            dm[i] = sharpen(y[i, 0].cpu().numpy())

        y = torch.cat((y, torch.from_numpy(dm).cuda()), 1)
        print("cat y.shape", y.shape)

        return SoftDiceLoss.forward(x, y)
