"""
Time:2020/8/7
Author:tzh666
Content:综合运用：通过迁移学习(使用预训练网络)解决cv问题
"""
# 1.导入模块
import torch

import time
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.utils.data as Data

# 2.加载需要分类的图象
data_dir = 'C:/Software/Pycharm/PycharmProjects/MachineStudyBase/Cat_Dog_data'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(data_dir+'/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir+'/test', transform=test_transforms)

train_loader = Data.DataLoader(train_data, batch_size=32)
test_loader = Data.DataLoader(test_data, batch_size=32)

# 3.对模型进行初始化
# 设定协议，当能够使用GPU时使用GPU，提高运算效率
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用预训练网络
model = models.densenet121(pretrained=True)

# 初始化时关闭模型参数的梯度跟踪
for param in model.parameters():
    param.requires_grad = False

# 预训练网络的分类器Classifier需要自己修改为对应的结构
classifier = nn.Sequential(nn.Linear(1024, 256),
                           nn.ReLU(),
                           nn.Dropout(0.2),
                           nn.Linear(256, 2),
                           nn.LogSoftmax(dim=1))

model.fc = classifier

print(model)

# 定义损失
criterion = nn.NLLLoss()

# 定义优化器
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)

# 采用GPU模式
model.to(device)

# 4.训练模型
# 所有数据训练循环一次
epochs = 1
steps = 0
# 训练五个批次测试一次
print_every = 5
running_loss = 0

for e in range(epochs):
    for images, labels in train_loader:
        steps += 1

        optimizer.zero_grad()

        # 使用GPU
        images, labels = images.to(device), labels.to(device)

        logps = model.forward(images)

        loss = criterion(logps, labels)

        loss.requires_grad = True

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # 5.测试模型
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()

            with torch.no_grad():
                for images, labels in test_loader:

                    images, labels = images.to(device), labels.to(device)

                    logps = model.forward(images)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss

                    # 计算精确度
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e+1}/{epochs}.. ",
                      f"Train loss: {running_loss / print_every:.3f}.. ",
                      f"Test loss: {test_loss / len(test_loader):.3f}.. ",
                      f"Test accuracy: {accuracy / len(test_loader):.3f}")

                running_loss = 0
                model.train()


# for device in ['cpu', 'cuda']:
#
#     criterion = nn.NLLLoss()
#     # Only train the classifier parameters, feature parameters are frozen
#     optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
#
#     model.to(device)
#
#     for ii, (inputs, labels) in enumerate(train_loader):
#
#         # Move input and label tensors to the GPU
#         inputs, labels = inputs.to(device), labels.to(device)
#
#         start = time.time()
#
#         outputs = model.forward(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         if ii == 3:
#             break
#
#     print(f"Device = {device}; Time per batch: {(time.time() - start) / 3:.3f} seconds")




