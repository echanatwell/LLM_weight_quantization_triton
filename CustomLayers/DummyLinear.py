import torch
import torch.nn as nn


class DummyLinear(nn.Module):
    def __init__(self, l, device):
        super(DummyLinear, self).__init__()
        self.weight = nn.Parameter(l.weight.clone()).to(device)

    def forward(self, x):
        return torch.matmul(x, self.weight.T)