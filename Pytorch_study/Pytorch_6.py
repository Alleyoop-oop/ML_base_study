"""
Time:2020/8/5
Author:tzh666
Content:训练神经网络
"""
import torch
import helper_own

from torch import nn
from torch import optim
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

# model[0]表示网络第一层，即Linear(784, 128)
# print(model[0])

# 定义负对数似然损失
criterion = nn.NLLLoss()

# 定义优化器用于更新权重和梯度
optimizer = optim.SGD(model.parameters(), lr=0.003)

# # 打印初始权重值
# print('Initial weights - ', model[0].weight)
#
# # 获取图象数据
# images, labels = next(iter(trainloader))
# images.resize_(64, 784)
#
# # 开始使梯度清零
# optimizer.zero_grad()
#
# output = model.forward(images)
# loss = criterion(output, labels)
#
# a = model[0].weight.grad
#
# # 计算梯度
# loss.backward()
# print('Gradient -', model[0].weight.grad)
#
# # 更新权重
# optimizer.step()
# print('Updated weights - ', model[0].weight)

epochs = 5
# 进行整个数据集的遍历，访问一次为1个周期(epoch)
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:

        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()

        output = model.forward(images)

        # 计算损失
        loss = criterion(output, labels)
        # 进行反向传播
        loss.backward()
        # 更新权重
        optimizer.step()

        running_loss += loss.item()

    print(f"Training loss: {running_loss/len(trainloader)}")

# 测试部分，选取数据集中的一个图象进行判断
images, labels = next(iter(trainloader))

img = images[0].view(1, 784)
# 关闭代码的梯度
with torch.no_grad():
    logps = model.forward(img)

# 计算预测的每种情况的概率
ps = torch.exp(logps)
helper_own.view_classify(img.view(1, 28, 28), ps)


