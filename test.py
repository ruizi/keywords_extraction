# -*- coding: utf-8 -*-
'''
@Author  : cai rui
@Date    : 2020/4/27 4:31 下午
'''
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data
from data_loader import DataManager


class MyData(data.Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]


def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data.unsqueeze(-1), data_length


if __name__ == '__main__':
    train_manager = DataManager(batch_size=2)  # batch数据提取器初始化构建

    # data = MyData(train_x)
    # print(data.data_seq)
    # data_loader = DataLoader(data, batch_size=2, shuffle=True,  # shuffle随机数种子，这里先关闭
    #                          collate_fn=collate_fn)
    #
    # batch_x, batch_x_len = iter(data_loader).next()
    # print(batch_x, batch_x_len)
    #
    # print("===")
    # for index, item in iter(data_loader):
    #     print(item)
    #
    # print("===")
    # batch_x_pack = rnn_utils.pack_padded_sequence(batch_x,
    #                                               batch_x_len, batch_first=True)
    #
    # net = nn.LSTM(1, 10, 2, batch_first=True)
    # h0 = torch.rand(2, 2, 10)
    # c0 = torch.rand(2, 2, 10)
    # out, (h1, c1) = net(batch_x_pack, (h0, c0))
    # out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
    # print('END')
