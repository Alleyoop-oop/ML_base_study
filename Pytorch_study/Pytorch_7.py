"""
Time:2020/8/6
Author:tzh666
Content:实战训练
"""
import torch
import helper_own
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F

# 数据规范化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# 下载训练集
trainset = datasets.FashionMNIST('./F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = Data.DataLoader(trainset, batch_size=64, shuffle=True)

# 下载测试集 train=False表示无法进行训练
testset = datasets.FashionMNIST('./F_MNIST_data/', download=True, train=False, transform=transform)
testloader = Data.DataLoader(testset, batch_size=64, shuffle=True)


# 定义神经网络
class Network(nn.Module):
    def __init__(self):
        # 隐藏层个数和输出形式可以自己设定
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # 为防止过拟合，采用dropout，按一定概率随机丢弃数据
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # 对输入扁平化
        x = x.view(x.shape[0], -1)

        # 前向传递
        x = self.fc1(x)
        x = self.fc(x)
        x = self.dropout(F.relu(x))
        # x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


# 定义模型
model = Network()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.003)
# 定义损失
criterion = nn.NLLLoss()

epochs = 30
train_losses, test_losses = [], []

# 训练模型training
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 验证模型validation
    else:
        test_loss = 0
        accuracy = 0

        # 关闭验证集的梯度，进行验证集的运算
        with torch.no_grad():
            # 设为验证模式，将丢弃率变为0
            model.eval()
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)

                # 获取概率最高的类别
                top_p, top_class = ps.topk(1, dim=1)

                # 比较正确值与判断值是否相同，注意：top_class与label维度不同，故需要将label转换为二维向量
                # 共64行，每行返回64个布尔值
                equals = top_class == labels.view(*top_class.shape)

                # 计算准确运算的百分比（即对equal求平均值）
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        # 还原为训练模式，存在dropout
        model.train()

        # 定义为列表形式，将每个epoch时的值储存下来，便于显示
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss / len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss / len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))


plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)

plt.show()


# 测试模型testing
model.eval()
images, labels = next(iter(testloader))
# 随机取库中的一个图象测试
# 需要转化成一维向量计算
img = images[0].view(1, 784)

with torch.no_grad():
    output = model.forward(img)

ps = torch.exp(output)

# 传入的图片为二维
helper_own.view_classify(img.view(1, 28, 28), ps, version="Fashion")














