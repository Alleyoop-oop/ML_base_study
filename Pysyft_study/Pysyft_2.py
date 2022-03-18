"""
Time:2020/8/18
Author:tzh666
Content:Pysyft 联邦学习
"""
import syft as sy
import torch
from torch import nn
from torch import optim

# 创建一对工作机
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# 创建一个玩具数据集
data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1.]], requires_grad=True)
target = torch.tensor([[0], [0], [1], [1.]], requires_grad=True)

# 通过以下方式获取每个工作机的训练数据的指针
data_bob = data[0:2]
target_bob = target[0:2]

data_alice = data[2:]
target_alice = target[2:]

# 向bob和alice发送一些训练数据
data_bob = data_bob.send(bob)
data_alice = data_alice.send(alice)
target_bob = target_bob.send(bob)
target_alice = target_alice.send(alice)

# 将指针组织到列表中
datasets = [(data_bob, target_bob), (data_alice, target_alice)]

# 构建一个玩具模型
model = nn.Linear(2, 1)

# 构建一个优化器
opt = optim.SGD(params=model.parameters(), lr=0.1)


# 将模型发送给各个节点（两个工作机），在这些节点上进行训练，然后拿回模型和梯度，在本地服务器上更新全局模型参数
def train():
    # 训练逻辑
    for e in range(10):
        # NEW）遍历每个工作人员的数据集
        for datas, targets in datasets:
            # NEW）将模型发送给对应的工作机
            model.send(datas.location)

            # 1) 消除之前的梯度（如果存在）
            opt.zero_grad()

            # 2) 预测
            pred = model(datas)

            # 3) 计算损失
            loss = ((pred - targets)**2).sum()

            # 4) 指出那些导致损失的参数（损失回传）
            loss.backward()

            # 5) 更新参数
            opt.step()

            # NEW）获取带梯度的模型和梯度
            model.get()
            loss = loss.get()

            # 6) 打印进程
            print(loss.data)


train()




