"""
Time:2020/8/21
Author:tzh666
Content:Pysyft 在加密数据上训练加密神经网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import syft as sy

# 一对工作机
hook = sy.TorchHook(torch)

alice = sy.VirtualWorker(id="alice", hook=hook)
bob = sy.VirtualWorker(id="bob", hook=hook)
james = sy.VirtualWorker(id="james", hook=hook)

# 一个数据集
data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1.]])
target = torch.tensor([[0], [0], [1], [1.]])


# 一个模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


model = Net()
optimizier = optim.SGD(params=model.parameters(), lr=0.1)

# 对模型和数据加密
data = data.fix_precision().share(bob, alice, crypto_provider=james, requires_grad=True)
target = target.fix_precision().share(bob, alice, crypto_provider=james, requires_grad=True)
model = model.fix_precision().share(bob, alice, crypto_provider=james, requires_grad=True)

# 用固定精度编码
opt = optimizier.fix_precision()

# print(data)

# 对模型进行训练
for e in range(20):
    opt.zero_grad()
    pred = model(data)
    loss = ((pred-target)**2).sum()
    loss.backward()
    opt.step()

    # 解密
    print(loss.get().float_precision())





