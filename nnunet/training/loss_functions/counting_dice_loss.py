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
        #print("\tx.shape:", x.shape)
        #print("\ty.shape:", y.shape)

        dm = np.empty(y.shape)
        idxs = y.shape[0]
        y_cpu = y.cpu().numpy()
        sums_gt = []
        for i in range(idxs):
            dm[i, 0] = sharpen(y_cpu[i, 0])
            sums_gt.append(np.sum(dm[i]))
        sums_gt = torch.tensor(sums_gt)
        #print("dm.shape:", dm.shape)

        y_ = torch.cat((y, torch.from_numpy(dm).cuda()), 1)
        #print("cat y_.shape", y_.shape)

        #x_cpu = x.cpu().numpy()
        sums_pred = torch.tensor([0 for _ in range(idxs)])
        for i in range(idxs):
            #sums_pred.append(np.sum(x_cpu[i]))
            sums_pred[i] = torch.sum(x[i, 1])

        ma_loss = 0.0001 * (sums_pred - sums_gt) ** 2
        print(sums_gt)
        print(sums_pred)

        return super(CountingDiceLoss, self).forward(x, y_) + ma_loss
