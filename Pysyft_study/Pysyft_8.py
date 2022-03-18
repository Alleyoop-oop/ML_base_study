"""
Time:2020/8/21
Author:tzh666
Content:Pysyft 对MNIST进行安全的训练和评估
  考虑您是服务器，并且您想对$n$个工作机持有的某些数据进行模型训练。服务器机密共享他的模型，
并将每个共享发送给工作机。工作机秘密共享他们的数据并在他们之间交换。在我们将要研究的配置
中，有2个工作机：alice和bob。交换共享后，他们每个人现在拥有自己的共享，另一工作机的数据
共享和模型共享。现在，计算可以开始使用适当的加密协议来私下训练模型。训练模型后，所有共享都
可以发送回服务器以对其进行解密。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import syft as sy
from torchvision import datasets, transforms
import time


# 定义超参数类（公开）
class Arguments:
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 64
        self.epochs = 10
        self.lr = 0.02
        self.seed = 1
        self.log_interval = 1
        self.precision_fractional = 3


args = Arguments()

# 为GPU设置种子
_ = torch.manual_seed(args.seed)

# 工作机的创建
hook = sy.TorchHook(torch)


def connect_to_workers(n_workers):
    return [
        sy.VirtualWorker(hook, id=f"worker{i+1}")
        for i in range(n_workers)
    ]


def connect_to_crypto_provider():
    return sy.VirtualWorker(hook, id="crypto_provider")


# 创建两个工作机
workers = connect_to_workers(n_workers=2)
# 创建一个安全节点
crypto_provider = connect_to_crypto_provider()

n_train_items = 640
n_test_items = 640


# # 创建私有数据加载器
# 假设MNIST数据集分布在各个部分中，每个部分都由我们的一个工作机持有。 然后，工作机将其数
# 据分批拆分，并在彼此之间秘密共享其数据。 返回的最终对象是这些秘密共享批次上的可迭代对象
def get_private_data_loaders(workers, crypto_provider):

    def one_hot_of(index_tensor):
        """
        Transform to one hot tensor
        将一维张量转化成二维，且对应的位置分别为1，其余为0

        Example:
            [0, 3, 9]
            =>
            [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]

        """
        onehot_tensor = torch.zeros(*index_tensor.shape, 10)  # 10 classes for MNIST
        onehot_tensor = onehot_tensor.scatter(1, index_tensor.view(-1, 1), 1)
        return onehot_tensor

    def secret_share(tensor):
        """
        Transform to fixed precision and secret share a tensor
        将数据变为私密
        """
        return (
            tensor.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)
        )

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./MNIST_data', train=True, download=True, transform=transformation),
        batch_size=args.batch_size
    )
    # 将训练集变为私有
    private_train_loader = [
        (secret_share(data), secret_share(one_hot_of(target)))
        for i, (data, target) in enumerate(train_loader)
        if i < n_train_items / args.batch_size
    ]
    # 加载测试集
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./MNIST_data', train=False, download=True, transform=transformation),
        batch_size=args.test_batch_size
    )
    # 将测试集变为私有
    private_test_loader = [
        (secret_share(data), secret_share(target.float()))
        for i, (data, target) in enumerate(test_loader)
        if i < n_test_items / args.test_batch_size
    ]

    return private_train_loader, private_test_loader


# 构建神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(args, model, private_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(private_train_loader):  # <-- now it is a private dataset
        start_time = time.time()

        optimizer.zero_grad()

        output = model(data)

        # loss = F.nll_loss(output, target)
        batch_size = output.shape[0]

        # 使用更为简单的均方误差
        loss = ((output - target) ** 2).sum().refresh() / batch_size

        loss.backward()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            loss = loss.get().float_precision()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s'.format(
                epoch, batch_idx * args.batch_size, len(private_train_loader) * args.batch_size,
                100. * batch_idx / len(private_train_loader), loss.item(), time.time() - start_time))


def test(args, model, private_test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in private_test_loader:
            start_time = time.time()

            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()

    correct = correct.get().float_precision()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct.item(), len(private_test_loader) * args.test_batch_size,
        100. * correct.item() / (len(private_test_loader) * args.test_batch_size)))


model = Net()
model = model.fix_precision().share(*workers, crypto_provider=crypto_provider, requires_grad=True)

optimizer = optim.SGD(model.parameters(), lr=args.lr)
optimizer = optimizer.fix_precision()

private_train_loader, private_test_loader = get_private_data_loaders(workers, crypto_provider)

for epoch in range(1, args.epochs + 1):
    train(args, model, private_train_loader, optimizer, epoch)
    test(args, model, private_test_loader)
