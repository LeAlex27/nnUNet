from nnunet.training.loss_functions.dice_loss import SoftDiceLoss


class CountingDiceLoss(SoftDiceLoss):
    def __init__(self, **sd_kw):
        super(CountingDiceLoss, self).__init__(**sd_kw)

    def forward(self, x, y, loss_mask=None):
        print("cdLoss:")
        print("\tx.shape:", x.shape)
        print("\ty.shape:", y.shape)
        return 0
