"""
Time:2020/8/19
Author:tzh666
Content:Pysyft 使用CNN在MNIST数据集上进行联邦学习
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy
import torch.utils.data as Data


# 设置学习任务（超参数的设定）
class Arguments:
    def __init__(self):
        # 训练时一批数据的大小
        self.batch_size = 64
        # 测试时一批数据的大小
        self.test_batch_size = 1000
        # 时刻数
        self.epochs = 10
        # 优化器参数
        self.lr = 0.01
        self.momentum = 0.5
        # GPU参数
        self.no_cuda = False
        self.seed = 1
        # 训练时输出损失的间隔
        self.log_interval = 30
        # 是否保存模型
        self.save_model = False


# 建立CNN网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


# 定义训练函数
def train(args, model, device, federated_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):  # <-- now it is a distributed dataset
        model.send(data.location)  # <-- NEW: send the model to the right location
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        model.get()  # <-- NEW: get the model back
        if batch_idx % args.log_interval == 0:
            loss = loss.get()  # <-- NEW: get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                100. * batch_idx / len(federated_train_loader), loss.item()))


# 定义测试函数
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # 初始化一对工作机
    hook = sy.TorchHook(torch)
    bob = sy.VirtualWorker(hook, id='bob')
    alice = sy.VirtualWorker(hook, id='alice')

    # 超参数类实例化
    args = Arguments()

    # GPU配置
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # 为当前GPU设置随机种子
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 将训练数据集转换为跨工作人员的联合数据集。
    # 现在，该联合数据集已提供给FederatedDataLoader。测试数据集保持不变
    federated_train_loader = sy.FederatedDataLoader(
        datasets.MNIST('./MNIST_data/', train=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])).federate((bob, alice)),
        batch_size=args.batch_size, shuffle=True)

    test_loader = Data.DataLoader(
        datasets.MNIST('./MNIST_data/', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # 构建模型和优化器
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # 进行训练和测试
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, federated_train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    # 保存模型
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
