import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
#保存方式1
torch.save(vgg16, "vgg16_method1.ph")

#保存方式2(官方推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.ph")

#陷阱
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()
torch.save(tudui, "tudui_method1.ph")