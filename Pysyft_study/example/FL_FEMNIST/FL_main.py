import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import syft as sy
import matplotlib.pyplot as plt


# 设置学习任务（超参数的设定）
class Arguments:
    def __init__(self):
        # 时刻数
        self.epochs = 150
        # 优化器参数
        self.lr = 0.05
        self.momentum = 0.5
        # 训练、测试一个批的个数
        self.train_batch_size = 50
        self.test_batch_size = 50
        # 判断进行哪种模式
        self.global_train = False
        self.federate_train = True
        self.single_train = True


# 建立CNN网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 62)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# 数据处理函数
def data_Init(args, node_1, node_2):

    # 加载FEMNIST输入数据
    X = torch.tensor(np.load('./new_data/train_data_x_3000.npy')).type(torch.FloatTensor).reshape(3000, 28, 28)
    # test_x = torch.tensor(np.load('./new_data/test_data_x_350.npy')).reshape(350, 28, 28)
    y = torch.tensor(np.load('./new_data/train_data_y_3000.npy'))
    # test_y = torch.tensor(np.load('./new_data/test_data_y_350.npy'))
    #
    # train_x = torch.tensor(np.load('./new_data/data_x_1000.npy')).type(torch.FloatTensor).reshape(1000, 28, 28)
    # train_y = torch.tensor(np.load('./new_data/data_y_1000.npy'))

    # # 生成2700个训练数据 300个测试数据
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, shuffle=True)

    train_dataset = Data.TensorDataset(train_x, train_y)
    test_dataset = Data.TensorDataset(test_x, test_y)

    # 将数据平均分到各个用户上，生成各个用户的数据集
    node1_train_dataset = Data.TensorDataset(train_x[:1350], train_y[:1350])
    node2_train_dataset = Data.TensorDataset(train_x[1350:], train_y[1350:])

    # 若为联邦学习模式，生成联邦训练数据集
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    node1_train_loader = Data.DataLoader(dataset=node1_train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    node2_train_loader = Data.DataLoader(dataset=node2_train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    FL_train_loader = sy.FederatedDataLoader(train_dataset.federate((node_1, node_2)), batch_size=args.train_batch_size, shuffle=True)

    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=2)

    return train_loader, node1_train_loader, node2_train_loader, FL_train_loader, test_loader


# 定义联邦学习训练函数
def FL_train(args, epoch, model, optimizer, federated_train_loader, device):
    model.train()

    running_loss = 0
    accuracy = 0
    for batch_idx, (data, target) in enumerate(federated_train_loader):

        # 将模型传递给对应用户
        model.send(data.location)
        data = data.to(device).unsqueeze(0).transpose(0, 1)
        target = target.long().to(device)

        optimizer.zero_grad()
        pred = model(data)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(pred, target)
        loss.backward()
        optimizer.step()

        # 本地收回模型
        model.get()

        # 准确率计算，需要先收回数据
        pred = pred.get().argmax(1, keepdim=True)
        accuracy += pred.eq(target.get().view_as(pred)).sum().item() / args.test_batch_size

        # 损失计算，需要先收回损失值
        loss = loss.get()
        running_loss += loss.item()

    print("epoch:{}, running_loss:{}, accuracy:{}".format(epoch, running_loss / len(federated_train_loader),
                                                              accuracy / len(federated_train_loader)))

    return running_loss / len(federated_train_loader), accuracy / len(federated_train_loader)


# 全局训练函数
def train(epoch, model, optimizer, train_loader, device):
    model.train()

    running_loss = 0
    accuracy = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device).unsqueeze(0).transpose(0, 1)
        target = target.long().to(device)

        optimizer.zero_grad()
        pred = model(data)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(pred, target)
        loss.backward()
        optimizer.step()

        # 准确率计算
        accuracy += accuracy_score(target.cpu(), pred.argmax(1, keepdim=True).view_as(target).cpu())

        running_loss += loss.item()

    print("epoch:{}, running_loss:{}, accuracy:{}".format(epoch, running_loss / len(train_loader),
                                                          accuracy / len(train_loader)))

    return running_loss / len(train_loader), accuracy / len(train_loader)


def test(epoch, model, test_loader, device):
    # 测试
    model.eval()

    test_loss = 0
    accuracy = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.type(torch.FloatTensor).to(device).unsqueeze(0).transpose(0, 1)
        target = target.long().to(device)
        pred = model(data)
        criteria = nn.CrossEntropyLoss()
        loss = criteria(pred, target)

        # 准确率计算
        accuracy += accuracy_score(target.cpu(), pred.argmax(1, keepdim=True).view_as(target).cpu())

        test_loss += loss.item()

    print("epoch:{}, test_loss:{}, accuracy:{}".format(epoch, test_loss / len(test_loader), accuracy / len(test_loader)))

    return test_loss / len(test_loader), accuracy / len(test_loader)


# 主函数
def main():
    args = Arguments()

    # GPU配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 联邦学习初始化
    hook = sy.TorchHook(torch)
    node_1 = sy.VirtualWorker(hook, id='node_1')
    node_2 = sy.VirtualWorker(hook, id='node_2')
    node_list = [node_1, node_2]

    # 初始数据设置
    train_loader, n1_train_loader, n2_train_loader, FL_train_loader, test_loader \
        = data_Init(args, node_1, node_2)

    # 存储列表初始化，用于显示
    GL_train_loss = []
    GL_train_acc = []
    GL_test_loss = []
    GL_test_acc = []
    FL_train_loss = []
    FL_train_acc = []
    FL_test_loss = []
    FL_test_acc = []
    SL_train_loss = []
    SL_train_acc = []
    SL_test_loss = []
    SL_test_acc = []

    if args.global_train:
        # /*******************GL-Learning*********************/
        # 构建模型和优化器
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            # 进行全局训练和测试
            train_loss, train_acc = train(epoch, model, optimizer, train_loader, device)
            test_loss, test_acc = test(epoch, model, test_loader, device)

            GL_train_loss.append(train_loss)
            GL_train_acc.append(train_acc)
            GL_test_loss.append(test_loss)
            GL_test_acc.append(test_acc)

    if args.federate_train:
        # /*******************FL-Learning*********************/
        # 构建模型和优化器
        model = Net().to(device)
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            # 进行联邦学习训练和测试
            train_loss, train_acc = FL_train(args, epoch, model, optimizer, FL_train_loader, device)
            test_loss, test_acc = test(epoch, model, test_loader, device)

            FL_train_loss.append(train_loss)
            FL_train_acc.append(train_acc)
            FL_test_loss.append(test_loss)
            FL_test_acc.append(test_acc)

    if args.single_train:
        # /*******************SL-Learning*********************/
        # 构建模型和优化器 2个用户
        model_list = [Net().to(device)] * len(node_list)
        optimizer_list = [optim.SGD(model_list[0].parameters(), lr=args.lr)] * len(node_list)
        dataloader_list = [n1_train_loader, n2_train_loader]
        for each_node in range(len(node_list)):
            SLn_train_loss = []
            SLn_train_acc = []
            SLn_test_loss = []
            SLn_test_acc = []
            for epoch in range(1, args.epochs + 1):
                train_loss, train_acc = train(
                    epoch, model_list[each_node], optimizer_list[each_node], dataloader_list[each_node], device)
                test_loss, test_acc = test(epoch, model_list[each_node], test_loader, device)

                SLn_train_loss.append(train_loss)
                SLn_train_acc.append(train_acc)
                SLn_test_loss.append(test_loss)
                SLn_test_acc.append(test_acc)

            SL_train_loss.append(SLn_train_loss)
            SL_train_acc.append(SLn_train_acc)
            SL_test_loss.append(SLn_test_loss)
            SL_test_acc.append(SLn_test_acc)


    # 显示图象
    x = []
    for i in range(args.epochs):
        x.append(i)

    # plt.plot(x, GL_train_acc, label='GL_train', linewidth=2.0)
    # plt.plot(x, GL_test_acc, label='GL_test', linewidth=2.0)
    plt.plot(x, FL_train_acc, label='FL_train', linewidth=2.0)
    plt.plot(x, FL_test_acc, label='FL_test', linewidth=2.0)
    plt.plot(x, SL_train_acc[0], label='SL1_train', linewidth=2.0)
    plt.plot(x, SL_test_acc[0], label='SL1_test', linewidth=2.0)
    plt.plot(x, SL_train_acc[1], label='SL2_train', linewidth=2.0)
    plt.plot(x, SL_test_acc[1], label='SL2_test', linewidth=2.0)

    plt.legend()  # 显示图例，如果注释改行，即使设置了图例仍然不显示
    plt.show()


if __name__ == '__main__':
    main()
