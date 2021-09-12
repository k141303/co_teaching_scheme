import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class ClassBalancedFocalCELoss(nn.Module):
    """
    ClassBalancedFocalCELoss is a loss function proposed in the following paper.
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
    """
    def __init__(self, class_counts, num_classes, beta=0.999, gamma=0, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average
        weights = [(1 - beta)/(1 - (beta ** class_counts[i])) for i in range(3)]
        self.weights = torch.tensor([w / sum(weights) * num_classes for w in weights])

    def forward(self, input, target):
        logpt = F.log_softmax(input,dim=-1)
        logpt = logpt.gather(1,target.view(-1, 1))
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * self.weights.to(input.device)[target.view(-1)] * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
