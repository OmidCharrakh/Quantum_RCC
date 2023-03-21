import numpy as np
import torch as th
import torch.nn as nn


class SHD_Loss(th.nn.Module):
    '''
    distance=SHD_Loss()(A_1=model(data), A_2=model(data))
    '''

    def __init__(self, double_for_anticausal=False):
        super(SHD_Loss, self).__init__()
        self.double_for_anticausal = double_for_anticausal

    def forward(self, A_1, A_2):
        A_1 = th.Tensor(A_1)
        A_2 = th.Tensor(A_2)
        diff = th.abs(A_1 - A_2)
        if self.double_for_anticausal:
            distance = th.sum(diff)
        else:
            diff = diff + diff.T
            diff[diff > 1] = 1
            distance = th.sum(diff)/2
        return distance


class CrossEntropyLoss_lst(th.nn.Module):

    def __init__(self):
        super(CrossEntropyLoss_lst, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, lst_pred, lst_true):
        distance = th.tensor(0)
        for outputs, target in zip(lst_pred, lst_true):
            distance = distance+self.loss(outputs, target)
        return distance


class MMDloss(th.nn.Module):

    def __init__(self, batch_size, bandwidths=[0.01, 0.1, 1, 10]):
        super(MMDloss, self).__init__()
        bandwidths = th.Tensor(bandwidths)
        s = th.cat([th.ones([batch_size, 1]) / batch_size, th.ones([batch_size, 1]) / -batch_size], 0)
        self.register_buffer('bandwidths', bandwidths.unsqueeze(0).unsqueeze(0))
        self.register_buffer('S', (s @ s.t()))

    def forward(self, x, y):
        X = th.cat([x, y], 0)
        XX = X @ X.t()
        X2 = (X * X).sum(dim=1).unsqueeze(0)
        exponent = -2*XX + X2.expand_as(XX) + X2.t().expand_as(XX)
        b = exponent.unsqueeze(2).expand(-1, -1, self.bandwidths.shape[2]) * - self.bandwidths
        lossMMD = th.sum(self.S.unsqueeze(2) * b.exp())
        return lossMMD
