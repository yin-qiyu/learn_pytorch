from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python 的用法 - 》tensor的数据类型
# 通过 transforms.ToTensor去解决两个问题
# 2、为什么需要Tensor数据类型


# 绝对路径 /Users/yinqiyu/PycharmProjects/learn_pytorch/data/train/ants_image/0013035.jpg
# 相对路径 data/train/ants_image/0013035.jpg
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("../logs")

# 1、 transform如何被使用(python)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# print(tensor_img)
writer.add_image("Tensor_img",tensor_img)
writer.close()
