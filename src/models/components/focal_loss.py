import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.ops.focal_loss import sigmoid_focal_loss
from kornia.losses import focal_loss


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction="mean"):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, output, target):
        output = output.squeeze(0)
        target = output.squeeze(0)
        loss = focal_loss(output, target, alpha=self.alpha,
                          gamma=self.gamma, reduction=self.reduction)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        clone = target.clone().to(dtype=torch.int64)
        # target = target.clone().detach().requires_grad_(True).to(dtype=torch.int64)
        # if input.dim() > 2:
        #     # N,C,H,W => N,C,H*W
        #     input = input.view(input.size(0), input.size(1), -1)
        #     input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
        #     input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        # target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, clone)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, clone.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
