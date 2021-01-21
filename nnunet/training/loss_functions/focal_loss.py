import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        print("inputs.size()", inputs.size())
        print("targets.size()", targets.size())
        BCE_loss = F.binary_cross_entropy_with_logits(inputs[:, 1:], targets, reduction='none')
        print("BCE_loss.size()", BCE_loss.size())
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        print("at.size()", at.size())
        pt = torch.exp(-BCE_loss)
        print("pt.size()", pt.size())
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
