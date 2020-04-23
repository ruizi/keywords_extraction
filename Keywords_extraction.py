# -*- coding: utf-8 -*-
'''
@Author  : cai rui
@Date    : 2020/4/19 11:47 下午
'''
import pickle

import torch
import torch.optim as optim
import json
import os

import yaml as yaml

from P_R_F1_calculator import f1_score, f1_score1
from data_manager import DataManager
from BiLSTM_CRF import BiLSTM_CRF


class keywords_extraction(object):
    def __init__(self, entry="train"):
        self.load_config()
        self._init_model(entry)

    # 载入配置文件
    def load_config(self):
        try:
            fopen = open("models/config.yml")
            config = yaml.load(fopen)
            fopen.close()
        except Exception as error:  # 载入失败使用默认配置文件
            print("Load config failed, using default config {}".format(error))
            fopen = open("models/config.yml", "w")
            config = {
                "embedding_size": 128,
                "hidden_size": 128,
                "batch_size": 20,
                "dropout": 0.5,
                "model_path": "models/",
                "tasg": ["ORG", "PER"]
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = config.get("model_path")
        self.dropout = config.get("dropout")

    # 初始化模型
    def _init_model(self, entry):
        if entry == "train":  # 如果是训练模式
            self.train_manager = DataManager(batch_size=self.batch_size)  # batch数据提取器初始化构建
            self.total_size = len(self.train_manager.batch_data)  # 一个epoch中要循环多少个batch就是batch的大小
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "vocab": self.train_manager.vocab,
                "tag_map": self.train_manager.tag_map,
            }
            self.save_params(data)  # 把相关参数做存储

            # dev板块
            dev_manager = DataManager(batch_size=128, data_type="dev")
            self.dev_batch = dev_manager.iteration()

            # 模型初始化定义
            self.model = BiLSTM_CRF(
                len(self.train_manager.vocab),
                self.train_manager.tag_map,
                self.embedding_size,
                self.hidden_size,
                self.batch_size,
                self.dropout,
            )

            # 先定义好模型，然后载入已经训练好的模型，如果存在
            self.restore_model()

        elif entry == "predict":
            data_map = self.load_params()
            input_size = data_map.get("input_size")
            self.tag_map = data_map.get("tag_map")
            self.vocab = data_map.get("vocab")
            self.model = BiLSTM_CRF(
                len(self.train_manager.vocab),
                self.train_manager.tag_map,
                self.embedding_size,
                self.hidden_size,
            )
            # self.restore_model()

        elif entry == "test":
            # dev板块
            test_manager = DataManager(batch_size=128, data_type="test")
            self.dev_batch = test_manager.iteration()
            # 模型初始化定义
            self.model = BiLSTM_CRF(
                len(self.train_manager.vocab),
                self.train_manager.tag_map,
                self.embedding_size,
                self.hidden_size,
                self.batch_size,
                self.dropout,
            )

            # 先定义好模型，然后载入已经训练好的模型，如果存在
            self.restore_model()

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "train20k_sep_taged_batch256.pkl"))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    def save_params(self, data):
        with open("models/data.pkl", "wb") as fopen:
            pickle.dump(data, fopen)

    def load_params(self):
        with open("models/data.pkl", "rb") as fopen:
            data_map = pickle.load(fopen)
        return data_map

    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.005, weight_decay=1e-4)  # 定义优化器，采用随机梯度下降
        for epoch in range(30):
            index = 0
            epoch_loss = 0
            for batch in self.train_manager.get_batch():  # 从data_manager中导入配置
                index += 1  # 训练位置
                self.model.zero_grad()  # 梯度初始化 需要要在每个batch计算之前清除梯度

                sentences, tags, length = zip(*batch)  # 从batch中拿到句子，目标标签和原始长度

                sentences_tensor = torch.tensor(sentences, dtype=torch.long)  # 三个list都转化为tensor
                tags_tensor = torch.tensor(tags, dtype=torch.long)
                length_tensor = torch.tensor(length, dtype=torch.long)

                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)  # 损失计算
                epoch_loss += loss
                print("""epoch [{}]  {}/{} \tloss {:.2f}""".format(epoch, index, self.total_size,
                                                                   loss.cpu().tolist()[0]))
                # 评估当前batch效果
                self.evaluate()
                # 反向传播优化
                loss.backward()
                optimizer.step()
                # 存储当前网络参数
            print("=======================================epoch_loss:", epoch_loss)
            torch.save(self.model.state_dict(), self.model_path + 'train20k_sep_taged_batch256.pkl')

    def evaluate(self):
        sentences, labels, length = zip(*self.dev_batch.__next__())  # 取得一个batch的训练数据
        with torch.no_grad():
            _, paths = self.model(sentences)  # 代入模型计算得到输出为paths
        # print("\tevaluate_this_batch")
        f1_score(labels, paths, self.model.tag_map)  # 代入计算函数

    def PRF1(self):
        for i in range(10):
            sentences, labels, length = zip(*self.dev_batch.__next__())  # 取得一个batch的训练数据
            # print(sentences[1])
            # print(labels[1])
            with torch.no_grad():
                _, paths = self.model(sentences)  # 代入模型计算得到输出为paths
            # print(paths[1])
            # print(labels[1])
            f1_score1(labels, paths, self.model.tag_map)  # 代入计算函数


if __name__ == "__main__":
    keywords = keywords_extraction("train")
    keywords.train()
    keywords.PRF1()
    keywords = keywords_extraction("test")
    keywords.PRF1()
