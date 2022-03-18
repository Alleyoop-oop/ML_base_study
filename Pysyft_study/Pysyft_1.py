"""
Time:2020/8/18
Author:tzh666
Content:Pysyft 基础
"""
import syft as sy
import torch

hook = sy.TorchHook(torch)

# 假设bob有一台机器(构建一个虚拟工作机)
bob = sy.VirtualWorker(hook, id='bob')

# 把张量发送给bob
x = torch.tensor([1, 2, 3, 4, 5])
y = torch.tensor([1, 1, 1, 1, 1])
# 保存为指向张量的指针形式
x_ptr = x.send(bob)
y_ptr = y.send(bob)
print(bob._objects)

# x_ptr.location : bob, location(位置)，对指针指向的位置的引用
# x_ptr.id_at_location : <random integer>, 张量存储在位置的id
# x_ptr.id : <random integer>, 指针张量的ID，它是随机分配的
# x_ptr.owner : "me", 拥有指针张量的工作机，这里是本地机器，名为“me”（我）
print(x_ptr)

# 收回我们的张量
x_ptr.get()
y_ptr.get()
print(bob._objects)

# 使用张量指针
x = torch.tensor([1, 2, 3, 4, 5]).send(bob)
y = torch.tensor([1, 1, 1, 1, 1]).send(bob)
# 不再是x和y在本地计算加法，而是将命令序列化并发送给Bob，由后者执行计算，创建张量z，然后将指向z的指针返回给我们
z = x + y
# 将结果返回到我们的机器上
z.get()

