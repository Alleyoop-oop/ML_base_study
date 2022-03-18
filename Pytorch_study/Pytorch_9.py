"""
Time:2020/8/7
Author:tzh666
Content:任意图象的加载
"""
from torchvision import datasets, transforms
import torch.utils.data as Data
import matplotlib.pyplot as plt
import helper_own

data_dir = 'C:/Software/Pycharm/PycharmProjects/MachineStudyBase/Cat_Dog_data'

# # 将图片的像素调整为255*255，比例缩放为224*224，并转换为张量形式
# transform = transforms.Compose([transforms.Resize(255),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor()])
#
# # 从存放图片的路径中加载图片
# dataset = datasets.ImageFolder(data_dir, transform=transform)
# dataloader = Data.DataLoader(dataset, batch_size=32, shuffle=True)
#
# # 随机显示图象
# images, labels = next(iter(dataloader))
# helper_own.imshow(images[0], normalize=False)
# plt.show()


# Define transforms for the training data and testing data
# 随机旋转、缩放/裁剪图像，然后翻转图像
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = Data.DataLoader(train_data, batch_size=32)
testloader = Data.DataLoader(test_data, batch_size=32)

# change this to the trainloader or testloader
data_iter = iter(testloader)

images, labels = next(data_iter)
helper_own.imshow(images[2], normalize=False)
plt.show()



