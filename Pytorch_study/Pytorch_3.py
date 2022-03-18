"""
Time:2020/8/4
Author:tzh666
Content:用Pytorch构建神经网络--用nn模块构建与上文件相同的网络
"""
from torch import nn
import torch.nn.functional as F


# 为网络创建类时，必须继承 nn.Module
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        # 创建一个线性转换模块𝑥𝐖+𝑏，其中有784个输入和256个输出
        self.hidden = nn.Linear(784, 256)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(256, 10)

        # Define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x


# 另一种方式 更简易
# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Inputs to hidden layer linear transformation
#         self.hidden = nn.Linear(784, 256)
#         # Output layer, 10 units - one for each digit
#         self.output = nn.Linear(256, 10)
#
#     def forward(self, x):
#         # Hidden layer with sigmoid activation
#         x = F.sigmoid(self.hidden(x))
#         # Output layer with softmax activation
#         x = F.softmax(self.output(x), dim=1)
#
#         return x


# 创建一个Network对象
model = Network()

# print(model)
for param in model.named_parameters():
    print(param)











