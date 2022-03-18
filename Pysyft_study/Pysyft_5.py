"""
Time:2020/8/19
Author:tzh666
Content:Pysyft 使用FederatedDataset进行联邦学习
"""
import torch as th
import syft as sy

# 创建一个沙箱Sandbox
sy.create_sandbox(globals(), verbose=False)

# 寻找数据集
boston_data = grid.search("#boston", "#data")
boston_target = grid.search("#boston", "#target")

# 加载模型和优化器
n_features = boston_data['alice'][0].shape[1]
n_targets = 1
model = th.nn.Linear(n_features, n_targets)

# 创建数据集
datasets = []
for worker in boston_data.keys():
    dataset = sy.BaseDataset(boston_data[worker][0], boston_target[worker][0])
    datasets.append(dataset)

dataset = sy.FederatedDataset(datasets)
print(dataset.workers)

# 创建优化器
optimizers = {}
for worker in dataset.workers:
    optimizers[worker] = th.optim.Adam(params=model.parameters(), lr=0.01)

# 将数据放入联邦数据加载器
train_loader = sy.FederatedDataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

# 进行训练
epochs = 50
for e in range(1, epochs+1):
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 传出模型
        model.send(data.location)
        # 设置优化器
        optimizer = optimizers[data.location.id]
        optimizer.zero_grad()

        pred = model(data)
        loss = ((pred.view(-1) - target)**2).mean()
        loss.backward()

        optimizer.step()

        model.get()

        loss = loss.get()
        running_loss += float(loss)

        if batch_idx % 8 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBatch loss: {:.6f}'.format(
                e, batch_idx, len(train_loader), 100*batch_idx/len(train_loader), loss.item()))

    print('Total loss', running_loss)
