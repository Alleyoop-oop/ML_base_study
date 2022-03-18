"""
Time:2020/8/19
Author:tzh666
Content:Pysyft 联邦学习
  使用高级聚合工具来允许参数由可信的“安全工作机”聚合，然后将最终结果模型发送回模型所有者（我们）
这样，只有安全工作机才能看到谁的模型参数来自谁。我们也许能够知道模型的哪些部分发生了更改，但是
我们不知道哪个工作人员（Bob或Alice）进行了哪些更改，从而创建了一层隐私。
"""
import torch
import syft as sy
from torch import nn, optim

# 1.创建数据所有者
# 初始化一对工作机
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id='bob')
alice = sy.VirtualWorker(hook, id='alice')
# 初始化一个安全机器
secure_worker = sy.VirtualWorker(hook, id='secure_worker')

# 构建数据集
data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1.]], requires_grad=True)
target = torch.tensor([[0], [0], [1], [1.]], requires_grad=True)

# 通过以下方式获取每个工作机的训练数据的指针
# 向bob和alice发送一些训练数据
data_bob = data[0:2].send(bob)
target_bob = target[0:2].send(bob)
data_alice = data[2:].send(alice)
target_alice = target[2:].send(alice)

# 2.构建模型
model = nn.Linear(2, 1)

print(model.weight)

iterations = 10
worker_iters = 5

# 多次迭代
for iters in range(iterations):
    # 3.将当前模型的副本发送给Alice和Bob，以便他们可以对自己的数据集执行学习步骤
    model_bob = model.copy().send(bob)
    model_alice = model.copy().send(alice)
    opt_bob = optim.SGD(params=model_bob.parameters(), lr=0.1)
    opt_alice = optim.SGD(params=model_alice.parameters(), lr=0.1)

    # 4.训练Alice和Bob的模型（并行）
    for wi in range(worker_iters):
        # 训练bob的模型
        # 梯度清零
        opt_bob.zero_grad()
        # 正向传递
        pred_bob = model_bob(data_bob)
        # 计算损失
        loss_bob = ((pred_bob-target_bob)**2).sum()
        # 反向求值
        loss_bob.backward()
        # 模型更新
        opt_bob.step()

        # 将梯度数据传回到本地计算机中
        loss_bob = loss_bob.get().data

        # 训练alice的模型
        # 梯度清零
        opt_alice.zero_grad()
        # 正向传递
        pred_alice = model_alice(data_alice)
        # 计算损失
        loss_alice = ((pred_alice-target_alice)**2).sum()
        # 反向求值
        loss_alice.backward()
        # 模型更新
        opt_alice.step()

        # 将梯度数据传回到本地计算机中
        loss_alice = loss_alice.get().data

        print("Bob:" + str(loss_bob) + " Alice:" + str(loss_alice))

    # 5.将两个更新的模型发送到安全工作机上
    model_bob.move(secure_worker)
    model_alice.move(secure_worker)

    # 6.将Bob和Alice的训练模型平均在一起，然后使用它来设置全局“模型”的值
    with torch.no_grad():
        model.weight.set_(((model_alice.weight.data+model_bob.weight.data)/2).get())
        model.bias.set_(((model_bob.bias.data+model_bob.bias.data)/2).get())

# 评估模型
preds = model(data)
loss = ((preds-target)**2).sum()
print(preds)
print(target)
print(loss.data)












