import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, Sequential
from torch.nn.modules.flatten import Flatten
from torch.utils.tensorboard import SummaryWriter


class Yqy(nn.Module):
    def __init__(self):
        super(Yqy, self).__init__()
        self.modul1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.modul1(x)
        return x

yqy = Yqy()
print(yqy)
input = torch.ones((64, 3, 32, 32))
output = yqy(input)
print(output.shape)

writer = SummaryWriter("../logs_seq")
writer.add_graph(yqy, input)
writer.close()