"""
Time:2020/8/19
Author:tzh666
Content:Pysyft 加密程序
"""
import time
import torch
import syft as sy
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data as Data


# 定义超参数类
class Arguments:
    def __init__(self):
        self.epochs = 2
        self.lr = 0.001
        self.test_batch_size = 1000
        self.batch_size = 64
        self.log_interval = 30
        self.seed = 1


# 定义训练网络(CNN)
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


# 定义训练函数
def train(train_distributed_dataset, optimizer, model, args, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_distributed_dataset):

        model.send(data.location)

        optimizer.zero_grad()
        pred = model(data)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()

        model.get()

        if batch_idx % args.log_interval == 0:
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_distributed_dataset) * args.batch_size,
                100. * batch_idx / len(train_distributed_dataset), loss.item()))
# def update(data, target, model, optimizer):
#     model.send(data.location)
#     optimizer.zero_grad()
#     pred = model(data)
#     loss = F.nll_loss(pred, target)
#     loss.backward()
#     optimizer.step()
#     return model
#
#
# def train(remote_dataset, compute_nodes, models, optimizers, params, bob, alice, james):
#     for data_index in range(len(remote_dataset[0]) - 1):
#         # 训练 更新模型
#         for remote_index in range(len(compute_nodes)):
#             data, target = remote_dataset[remote_index][data_index]
#             models[remote_index] = update(data, target, models[remote_index], optimizers[remote_index])
#
#         # 加密聚合
#         new_params = list()
#         for param_i in range(len(params[0])):
#             spdz_params = list()
#             for remote_index in range(len(compute_nodes)):
#                 # 从每个工作机中选择相同的参数并复制
#                 copy_of_parameter = params[remote_index][param_i].copy()
#
#                 # 由于SMPC只能使用整数（不能使用浮点数）
#                 # 因此我们需要使用Integers存储十进制信息。
#                 # 换句话说，我们需要使用“固定精度”编码。
#                 fixed_precision_param = copy_of_parameter.fix_precision()
#
#                 # 现在我们在远程计算机上对其进行加密。
#                 # 注意，fixed_precision_param“已经”是一个指针。
#                 # 因此，当我们调用share时，它实际上是对指向的数据进行加密。
#                 # 而它会返回一个指向MPC秘密共享对象的指针，也就是我们需要的共享分片。
#
#                 encrypted_param = fixed_precision_param.share(bob, alice, crypto_provider=james)
#
#                 # 现在我们获取指向MPC共享值的指针
#                 param = encrypted_param.get()
#
#                 # 保存参数，以便我们可以使用工作机的相同参数取平均值
#                 spdz_params.append(param)
#
#             new_param = (spdz_params[0] + spdz_params[1]).get().float_precision() / 2
#             new_params.append(new_param)
#
#         # cleanup
#         with torch.no_grad():
#             for model in params:
#                 for param in model:
#                     param *= 0
#
#             for model in models:
#                 model.get()
#
#             for remote_index in range(len(compute_nodes)):
#                 for param_index in range(len(params[remote_index])):
#                     params[remote_index][param_index].set_(new_params[param_index])


# 定义测试函数
def test(models, test_loader):
    models.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = models(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 主函数
def main():
    # 加载数据
    args = Arguments()
    train_loader, test_loader = data_download(args)

    # 创建工作机
    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id='bob')
    alice = sy.VirtualWorker(hook, id='alice')
    james = sy.VirtualWorker(hook, id='james')
    compute_nodes = [bob, alice]

    # 将数据发送给工作机(手动发送)
    train_distributed_dataset = []
    # remote_dataset = (list(), list())

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.send(compute_nodes[batch_idx % len(compute_nodes)])
        target = target.send(compute_nodes[batch_idx % len(compute_nodes)])
        train_distributed_dataset.append((data, target))
        # remote_dataset[batch_idx % len(compute_nodes)].append((data, target))

    model = Network()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # bobs_model = Network()
    # alices_model = Network()

    # bobs_optimizer = optim.SGD(bobs_model.parameters(), lr=args.lr)
    # alices_optimizer = optim.SGD(alices_model.parameters(), lr=args.lr)

    # models = [bobs_model, alices_model]
    # params = [list(bobs_model.parameters()), list(alices_model.parameters())]
    # optimizers = [bobs_optimizer, alices_optimizer]

    t = time.time()

    # 进行训练和测试
    for epoch in range(1, args.epochs + 1):
        train(train_distributed_dataset, optimizer, model, args, epoch)
        test(model, test_loader)

    total_time = time.time() - t
    print('Total', round(total_time, 2), 's')


if __name__ == '__main__':
    main()








