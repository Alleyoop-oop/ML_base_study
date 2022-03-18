"""
Time:2020/8/4
Author:tzh666
Content:Pytorch 基础
"""
import torch


def activation(x):
    """ Sigmoid activation function
        S式模型
           Arguments
           ---------
           x: torch.Tensor
       """
    return 1/(1+torch.exp(-x))


# 1.简单学习网络
def function1():
    # 生成初始数据
    torch.manual_seed(7)  # 为CPU设置种子用于生成随机数

    # 创建特征值x(1 row 3 column)
    features = torch.randn(1, 3)
    # 随机创建权重值w，与x结构相同
    weights = torch.randn_like(features)
    # 随机创建偏差值bias
    bias = torch.randn(1, 1)

    # 计算简单网络输出
    output = activation(torch.sum(features*weights)+bias)  # 向量各元素直接相乘并求和
    # output = activation(torch.mm(features, weights.view(3, 1))+bias)  # 向量做内积，采用view将w变为3行一列

    print(weights.shape)
    print(output)


# 2.带隐藏层的学习网络
def function2():
    # 生成初始数据
    torch.manual_seed(7)
    # 创建特征值x(1 row 3 column)
    features = torch.randn(1, 3)

    # 设计各层元素个数
    n_input = features.shape[1]
    n_hidden = 2
    n_output = 1

    # 创建权重W1、W2（一层一个）
    W1 = torch.randn(n_input, n_hidden)
    W2 = torch.randn(n_hidden, n_output)

    # 创建偏差值B1、B2
    B1 = torch.randn(1, n_hidden)
    B2 = torch.randn(1, n_output)

    # 计算隐藏层的值H
    H = activation(torch.mm(features, W1)+B1)

    # 计算网络输出Y
    Y = activation(torch.mm(H, W2)+B2)

    print(Y)


# function1()
function2()


