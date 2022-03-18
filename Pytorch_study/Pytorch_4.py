"""
Time:2020/8/4
Author:tzh666
Content:前两例结合
"""
import numpy as np
import helper_own
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

import torch.utils.data as Data
# 导入torchvision中的数据集和规则
from torchvision import datasets, transforms

# 定义一个规则使得数据规范化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
# 下载MNIST数据集
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
# 将数据加载到trainloader中
trainloader = Data.DataLoader(trainset, batch_size=64, shuffle=True)


# 定义神经网络类
class Network(nn.Module):
    # 输入层有784个单元，然后是有128个单元的隐藏层和一个ReLU激活函数
    # 接着是有64个单元的隐藏层和一个ReLU激活函数，最终是一个有10个单元的应用softmax激活函数的输出层
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)

        return x


# 实例化模型
model = Network()
# 初始化权重和偏差
model.fc1.bias.data.fill_(0)
model.fc1.weight.data.normal_(std=0.01)

# 前向传递，传入图像
dataiter = iter(trainloader)
images, labels = dataiter.next()
# 将图片存储为一维形式
images.resize_(images.shape[0], 1, 784)

# 通过网络前向传递图片
img_idx = 0
ps = model.forward(images[img_idx, :])

img = images[img_idx]
helper_own.view_classify(img.view(1, 28, 28), ps)





