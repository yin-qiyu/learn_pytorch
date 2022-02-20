from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter("logs")
img = Image.open("images/3.png")
print(img)

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)
# writer.close()

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)


# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
# img PIL -> img_resize PIL
img_resize = trans_resize(img)
print(img_resize)


# img_resize PIL -> totensor -> img_reasize tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)


#Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(128, 60)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)


writer.close()