import torch
from torch import nn


class Yqy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        ouput = input + 1
        return ouput


yqy = Yqy()
x = torch.tensor(1.0)
outut = yqy(x)
print(outut)
