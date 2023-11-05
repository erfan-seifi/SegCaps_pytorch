import torch.nn.functional as F
import torch


def compute_loss(output, target):
    loss = (target * F.relu(0.9 - output) + 0.2 * (1 - target) * F.relu(output - 0.1)).mean()
    return loss


def compute_loss_adaptive_ratio(output, target):
    h = w = 512
    target_ratio = torch.sum(target==1.0, dim=[1, 2], keepdim=True) / (h * w)
    loss = (target * F.relu(0.9 - output) + target_ratio * (1 - target) * F.relu(output - 0.1)).mean()
    return loss


def compute_loss_balanced(output, target):
    h = w = 512
    target_ratio = torch.sum(target==1.0, dim=[1, 2], keepdim=True) / (h * w)
    balance_ratio = (1 + 0.04) / (target_ratio + 0.04)
    loss = (balance_ratio * target * F.relu(0.9 - output) + target_ratio * (1 - target) * F.relu(output - 0.1))
    loss = (loss * 1e+4).mean()
    return loss
