"""
Time:2020/8/5
Author:tzh666
Content:损失和梯度的计算
"""
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.utils.data as Data

# 定义一个规则使得数据规范化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
# 下载MNIST数据集
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
# 将数据加载到trainloader中
trainloader = Data.DataLoader(trainset, batch_size=64, shuffle=True)

# 定义神经网络类
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      # 使用log-softmax输出
                      nn.LogSoftmax(dim=1))

# 定义交叉熵损失
# criterion = nn.CrossEntropyLoss()
# 这里使用负对数似然损失
criterion = nn.NLLLoss()

# 获取图象数据
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
# 向前传递，得到分数值
logps = model(images)

print(logps)
print(labels)
# 计算损失(传入网络输出和正确标签)
loss = criterion(logps, labels)

print(loss)

print('Before backward pass: \n', model[0].weight.grad)

# 计算loss对第一层权重w1的梯度值
loss.backward()
print('After backward pass: \n', model[0].weight.grad)













