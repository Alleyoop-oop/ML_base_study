"""
Time:2020/8/22
Author:tzh666
Content:Pysyft 小测试：采用fedavg算法实现多个用户的联邦学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
import numpy as np
import torch.utils.data as Data


# 定义超参数类
class Arguments:
    def __init__(self):
        # 用户数量
        self.n_workers = 2
        # 一次训练选取的用户数量
        self.K = 2
        # 每一次训练的轮次
        self.E = 1
        # 总训练时期
        self.epochs = 2
        self.lr = 0.001
        self.test_batch_size = 1000
        self.batch_size = 64
        self.log_interval = 30


# 构建神经网络
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 数据加载函数
def data_download(args):
    train_loader = Data.DataLoader(
        datasets.MNIST('./MNIST_data/', train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=args.batch_size, shuffle=True)

    test_loader = Data.DataLoader(
        datasets.MNIST('./MNIST_data/', train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), batch_size=args.test_batch_size, shuffle=True)

    return train_loader, test_loader


# 创建用户
def get_workers(n_workers):
    hook = sy.TorchHook(torch)
    return [
        sy.VirtualWorker(hook, id=f"worker{i+1}")
        for i in range(n_workers)
    ]


# 发送数据到用户
def data_send(train_loader, compute_nodes):
    train_distributed_dataset = []
    # 此处将数据平均发送给用户
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.send(compute_nodes[batch_idx % len(compute_nodes)])
        target = target.send(compute_nodes[batch_idx % len(compute_nodes)])
        train_distributed_dataset.append((data, target))

    return train_distributed_dataset


# 随机选取K个用户
def select_clients(workers, K, seed=1):
    np.random.seed(seed)
    return np.random.choice(workers, K, replace=False).tolist()


# 定义训练函数
def train(args, model, train_loader, epoch, workers):
    model.train()

    # 进行E次用户的训练
    for e in range(1, args.E+1):
        running_loss = 0

        # 随机选取K个用户
        # workers_used = select_clients(workers, args.K)

        # 构建这些用户的模型和优化器
        models = []
        optimizers = []
        for k in range(len(workers)):
            model_worker = model.copy()
            models.append(model_worker)
            optimizer_worker = optim.SGD(params=model_worker.parameters(), lr=args.lr)
            optimizers.append(optimizer_worker)

        # 将数据发送给选中用户
        train_distributed_dataset = data_send(train_loader, workers)

        # 对选中用户里的第order个用户进行训练
        for batch_idx, (data, target) in enumerate(train_distributed_dataset):
            # data = data.view(data.shape[0], -1)
            # 得到此时对应训练的模型序号
            order = batch_idx % len(workers)
            # 将模型发送给对应的用户
            models[order].send(data.location)

            optimizers[order].zero_grad()
            pred = models[order](data)
            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizers[order].step()

            models[order].get()
            # loss = loss.get()

            # print('batch: {}/, loss: {}'.format(batch_idx, loss.item()))
            if batch_idx % args.log_interval == 0:
                loss = loss.get()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * args.batch_size, len(train_distributed_dataset) * args.batch_size,
                    100. * batch_idx / len(train_distributed_dataset), loss.item()))

                running_loss += loss.item()

        # 模型平均，更新全局模型
        with torch.no_grad():
            # 计算更新后的模型参数
            model_weight = 0
            model_bias = 0
            for k in range(len(workers)):
                model_weight += models[k].fc1.weight.data
                model_bias += models[k].fc1.bias.data

            print(model_weight)
            model.fc1.weight.set_(model_weight / args.K)
            model.fc1.bias.set_(model_bias / args.K)

        print('\nTrain set: epoch: {}, turn: {}, train loss: {:.4f}'.format(epoch, e, running_loss))


# 定义测试函数
def test(model, test_loader):
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, accuracy, len(test_loader.dataset),
        100. * accuracy / len(test_loader.dataset)))


# 主函数
def main():
    # 加载数据
    args = Arguments()
    train_loader, test_loader = data_download(args)

    # 创建工作用户
    workers = get_workers(args.n_workers)
    # print(workers)

    # 构建模型和优化器
    model = Network()
    # print(model.weight)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        # 在各个用户上训练模型
        train(args, model, train_loader, epoch, workers)

        # 在本地进行测试模型
        test(model, test_loader)


if __name__ == '__main__':
    main()
