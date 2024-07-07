import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
        
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, cross_entropy_weight=0.5, focal_loss_weight=0.5, reduction='mean'):
        super(HybridLoss, self).__init__()
        self.cross_entropy_weight = cross_entropy_weight
        self.focal_loss_weight = focal_loss_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        fl_loss = self.focal_loss(inputs, targets)
        return self.cross_entropy_weight * ce_loss + self.focal_loss_weight * fl_loss