"""
Time:2020/8/6
Author:tzh666
Content:保存和加载模型
"""
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import helper_own
# 用于训练和创建网络
import fc_model
import torch.utils.data as Data

# 数据规范化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# 下载训练集
trainset = datasets.FashionMNIST('./F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = Data.DataLoader(trainset, batch_size=64, shuffle=True)

# 下载测试集 train=False表示无法进行训练
testset = datasets.FashionMNIST('./F_MNIST_data/', download=True, train=False, transform=transform)
testloader = Data.DataLoader(testset, batch_size=64, shuffle=True)


# 加载字典
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])

    return model


# 创建网络(输入个数，输出个数，隐藏层中数据个数)
model = fc_model.Network(784, 10, [512, 256, 128])
# model = load_checkpoint('checkpoint.pth')

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
fc_model.train(model, trainloader, testloader, criterion, optimizer, epochs=2)

# PyTorch网络的参数保存在模型的state_dict中。可以看到这个状态字典包含每个层级的权重和偏差矩阵。
print("Our model: \n\n", model, '\n')
print("The state dict keys: \n\n", model.state_dict().keys())

# checkpoint = {'input_size': 784,
#               'output_size': 10,
#               'hidden_layers': [each.out_features for each in model.hidden_layers],
#               'state_dict': model.state_dict()}
#
# # 保存字典到checkpoint.pth文件中
# torch.save(checkpoint, 'checkpoint.pth')







