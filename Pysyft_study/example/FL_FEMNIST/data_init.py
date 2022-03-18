import json
import numpy as np
import os
from collections import defaultdict
import logging
logging.basicConfig(level=logging.INFO, filename="mylog.log", filemode='w', format="%(filename)s[line:%(lineno)d] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# 加载FEMNIST数据集
def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    """
    parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def main():
    train_data_dir = './ori_data/train'
    test_data_dir = './ori_data/test'
    all_data_dir = './ori_data/all_data'

    # users:380个
    # train_data:每个user有对应个数的训练数据，每个数据结构为(784,)
    # test_data:每个user有对应个数的测试数据，每个数据结构为(784,)
    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    # 取多个用户的训练数据，并合并，得到3000个训练数据和350个测试数据
    all_train_data_images = []
    all_train_data_labels = []
    all_test_data_images = []
    all_test_data_labels = []
    for i in range(11):
        all_train_data_images += train_data[users[i]]['x']
        all_test_data_images += test_data[users[i]]['x']
        all_train_data_labels += train_data[users[i]]['y']
        all_test_data_labels += test_data[users[i]]['y']

    all_train_data_images = all_train_data_images[:1000]
    all_test_data_images = all_test_data_images[:350]
    all_train_data_labels = all_train_data_labels[:1000]
    all_test_data_labels = all_test_data_labels[:350]

    # np.save('./new_data/train_data_x_3000.npy', all_train_data_images)
    # np.save('./new_data/test_data_x_350.npy', all_test_data_images)
    # np.save('./new_data/train_data_y_3000.npy', all_train_data_labels)
    # np.save('./new_data/test_data_y_350.npy', all_test_data_labels)
    #
    # np.save('./new_data/data_x_1000.npy', all_train_data_images)
    # np.save('./new_data/data_y_1000.npy', all_train_data_labels)



if __name__ == '__main__':

    main()
