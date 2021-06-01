import numpy as np
import pickle
import skimage
import torch
from skimage.morphology import label
from skimage.measure import regionprops
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss
from nnunet.utilities.nd_softmax import softmax_helper
import matplotlib.pyplot as plt


class CountingDiceLoss(torch.nn.Module):
    def __init__(self, label_loss, density_map_loss, count_loss, output_folder=None):
        super(CountingDiceLoss, self).__init__()
        self.loss = SoftDiceLoss(softmax_helper, **{'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
        self.loss_density_map = torch.nn.MSELoss()  # WeightedRobustCrossEntropyLoss([0.001, 0.999])

        self.loss_n_ma = torch.nn.MSELoss()
        self.n = 0
        self.output_folder = output_folder

        self.label_loss = label_loss
        self.density_map_loss = density_map_loss
        self.count_loss = count_loss

        self.l_ = []
        self.l_dm = []
        self.l_n = []
        self.l_total = []
        self.sizes = []

    def __del__(self):
        if self.output_folder is not None:
            with open(self.output_folder + '/losses.pickle', 'wb') as f:
                pickle.dump(self.l_, f)
                pickle.dump(self.l_dm, f)
                pickle.dump(self.l_n, f)
                pickle.dump(self.l_total, f)
                pickle.dump(self.sizes, f)

    def forward(self, x, y, loss_mask=None):
        # create gt density map
        y_cpu = y.cpu().numpy()
        dm = np.empty_like(y_cpu[:, 0:1])
        for i in range(y.shape[0]):
            dm[i, 0] = self.sharpen(y_cpu[i, 0])
            # dm[i, 0] = self.pixel(y_cpu[i, 0])
        # self.save_img(dm, '/cluster/husvogt/debug_imgs/{:04d}_{:03d}.png')
        dm = torch.from_numpy(dm).cuda()
        y_n_ma = torch.sum(dm)
        x_n_ma = torch.sum(x[:, 2])  # -1: = 3:

        l_ = self.loss(x[:, :2], y)
        l_dm = self.loss_density_map(x[:, 2:], dm)
        l_n = self.loss_n_ma(x_n_ma, y_n_ma)
        l_total = torch.tensor(0).cuda()
        if self.label_loss:
            l_total += l_
        if self.density_map_loss:
            l_total += l_dm
        if self.count_loss:
            l_total += l_n

        print("total loss:", l_total)

        self.l_.append(l_.detach().cpu().numpy())
        self.l_dm.append(l_dm.detach().cpu().numpy())
        self.l_n.append(l_n.detach().cpu().numpy())
        self.l_total.append(l_.detach().cpu().numpy())
        self.sizes.append(list(x.size()))

        return l_total

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

    @staticmethod
    def pixel(img):
        t = np.zeros_like(img)
        _, props = CountingDiceLoss.labels_and_props(img)

        for p in props:
            p_i = int(p.centroid[0])
            p_j = int(p.centroid[1])
            t[p_i, p_j] = 1

        return t
