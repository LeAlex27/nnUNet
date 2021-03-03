from nnunet.training.loss_functions.dice_loss import SoftDiceLoss


class CountingDiceLoss(SoftDiceLoss):
    def __init__(self, **sd_kw):
        super(CountingDiceLoss, self).__init__(**sd_kw)

    def forward(self, x, y, loss_mask=None):
        return 0
