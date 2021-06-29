import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import *
device = use_cuda()

class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=0.5, reduction='mean'):
        nn.Module.__init__(self)
        self.weight = torch.FloatTensor([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            #weight=self.weight,
            reduction=self.reduction
        )
    
    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=42, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    

class F1Loss(nn.Module):
    def __init__(self, classes=42, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

    
class BlendedLoss(nn.Module):
    def __init__(self, classes=18, epsilon=1e-7, smoothing=0.0, dim=-1):
        super().__init__()
        ## F1Loss
        self.classes = classes
        self.epsilon = epsilon
        ## LabelSmoothingLoss
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, y_pred, y_true):
        y_pred = y_pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(y_pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, y_true.data.unsqueeze(1), self.confidence)
        labelsmoothingloss = torch.mean(torch.sum(-true_dist * y_pred, dim=self.dim))
    
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        f1loss = 1 - f1.mean()
        
        return f1loss * 0.8 + labelsmoothingloss * 0.2
    

'''
Mix Cross Entropy Loss & Label Smoothing Loss
'''
class CELSLoss(nn.Module):
    def __init__(self, classes=42, smoothing=0.0, dim=-1):
        super(CELSLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        ce_loss =  F.cross_entropy(pred, target)
        
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        ls_loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        
        return ce_loss + ls_loss