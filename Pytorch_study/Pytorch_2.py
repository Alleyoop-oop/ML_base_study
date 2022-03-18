"""
Time:2020/8/4
Author:tzh666
Content:用Pytorch构建神经网络
"""
import torch
import matplotlib.pyplot as plt

import torch.utils.data as Data
# 导入torchvision中的数据集和规则
from torchvision import datasets, transforms

# 定义一个规则使得数据规范化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
# 下载MNIST数据集
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
# 将数据加载到trainloader中
trainloader = Data.DataLoader(trainset, batch_size=64, shuffle=True)

# 变为迭代器，循环访问数据集以进行训练
dataiter = iter(trainloader)
images, labels = dataiter.next()

# print(images)
# 每批有 64 个图像，图像有 1 个颜色通道，共有 28x28 个图像
print(type(images))
print(images.shape)
print(labels.shape)

# 显示该图象 其中images的下标代表取到的这一批次中的图象位置
# 每一次的图象不同，因为迭代器进行到了下一批
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
plt.title('%i' % labels[1])
plt.show()


# 构建多层网络
def activation(x):
    return 1/(1+torch.exp(-x))


# 初始化网络参数
n_input = 784
n_hidden = 256
n_output = 10

inputs = images.view(images.shape[0], -1)

w1 = torch.randn(n_input, n_hidden)
w2 = torch.randn(n_hidden, n_output)
b1 = torch.randn(n_hidden)
b2 = torch.randn(n_output)

h = activation(torch.mm(inputs, w1)+b1)

output = torch.mm(h, w2)+b2


# 采用softmax函数计算概率分布
def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)


probabilities = softmax(output)

print(probabilities.shape)
# 对列进行求和，验证是否概率和为1
print(probabilities.sum(dim=1))






