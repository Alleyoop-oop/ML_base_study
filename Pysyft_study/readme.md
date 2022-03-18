**此文件夹为pysyft联邦学习框架的学习**

# requirements:

python3.7

syft == 0.2.4

torch == 1.4.0 + cu92

# install：

pip install torch

pip install syft

# introduction:

pysyft1: pysyft 基础知识，创建工作机、数据的传输。

pysyft2: Pysyft 进行简单的联邦学习，两个工作机。

pysyft3: 使用高级聚合工具来允许参数由可信的“安全工作机”聚合，然后将最终结果模型发送回模型所有者（我们），只有安全工作机才能看到谁的模型参数来自谁。

pysyft4: Pysyft 使用CNN在MNIST数据集上进行联邦学习。

pysyft5: Pysyft 使用FederatedDataset进行联邦学习。

pysyft6: Pysyft 加密程序。

pysyft7: Pysyft 在加密数据上训练加密神经网络。

pysyft8: Pysyft 对MNIST进行安全的训练和评估。

pysyft_test: 采用fedavg算法实现多个用户的联邦学习。



