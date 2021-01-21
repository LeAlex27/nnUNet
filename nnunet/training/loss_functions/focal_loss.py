import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1.0 - alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        # BCE(p, y) = -log(pt) => pt = exp(-BCE(p, y))
        #      loss = -log(pt) * at(1-pt)**g
        # =>        = BCE(p, y) * at * (1 - pt)**g

        BCE_loss = F.binary_cross_entropy_with_logits(inputs[:, 1:], targets, reduction='none')

        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.view(-1)).view(targets.size())
        pt = torch.exp(-BCE_loss)

        F_loss = BCE_loss * at * (1 - pt) ** self.gamma

        return F_loss.mean()
