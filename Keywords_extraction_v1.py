# -*- coding: utf-8 -*-
"""
@Author  : cai rui
@Date    : 2020/4/28 2:30 下午
"""

import pickle
import torch
import torch.optim as optim
import yaml as yaml

from P_R_F1_calculator import f1_score, f1_score1
from data_loader import DataManager
from BiLSTM_CRF_test import BiLSTM_CRF


def load_params():
    with open("models/data.pkl", "rb") as data_map_file:
        data_map = pickle.load(data_map_file)
    return data_map


def save_params(data):
    with open("models/data.pkl", "wb") as data_map_file:
        pickle.dump(data, data_map_file)


class keywords_extraction(object):
    # 参数初始化
    def __init__(self, entry="train"):
        # 先从config文件中导入模型配置参数
        try:
            config_file = open("models/config.yml")
            config = yaml.load(config_file, Loader=yaml.FullLoader)
            config_file.close()
        except Exception as error:  # 载入失败使用默认配置文件
            print("导入配置文件失败，将使用默认配置".format(error))
            config_file = open("models/config.yml", "w")
            config = {
                "embedding_size": 128,
                "hidden_size": 128,
                "batch_size": 20,
                "dropout": 0.5,
                "model_path": "models/",
            }
            yaml.dump(config, config_file)
            config_file.close()
        self.embedding_size = config.get("embedding_size")
        self.hidden_size = config.get("hidden_size")
        self.batch_size = config.get("batch_size")
        self.model_path = config.get("model_path")
        self.dropout = config.get("dropout")

        # 对模型初始化
        self._init_model(entry)

    # 初始化模型
    def _init_model(self, entry):
        if entry == "train":  # 如果是训练模式
            self.train_manager = DataManager(batch_size=self.batch_size, data_type="train")  # batch数据提取器初始化构建
            self.total_size = len(self.train_manager.sentences_loader)  # 一个epoch中要循环多少个batch就是batch的大小
            data = {
                "batch_size": self.train_manager.batch_size,
                "input_size": self.train_manager.input_size,
                "vocab": self.train_manager.vocab,
                "tag_map": self.train_manager.tag_map,
            }
            save_params(data)  # 把相关参数做存储

            # dev板块
            dev_manager = DataManager(batch_size=self.batch_size, data_type="dev")
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
            data_map = load_params()
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
            self.test_manager = DataManager(batch_size=128, data_type="test")
            self.dev_batch = self.test_manager.iteration()
            # 模型初始化定义
            self.model = BiLSTM_CRF(
                len(self.test_manager.vocab),
                self.test_manager.tag_map,
                self.embedding_size,
                self.hidden_size,
                self.batch_size,
                self.dropout,
            )

            # 先定义好模型，然后载入已经训练好的模型，如果存在
            self.restore_model()

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "train500_sep_taged_batch128.pkl"))
            print("model restore success!")
        except Exception as error:
            print("model restore faild! {}".format(error))

    def train(self):
        # optimizer = optim.SGD(self.model.parameters(), lr=0.005, weight_decay=1e-4)  # 定义优化器，采用随机梯度下降
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        for epoch in range(30):
            index = 0
            epoch_loss = 0
            for batch in self.train_manager.get_batch():  # 从data_manager中导入配置
                index += 1  # 训练位置
                self.model.zero_grad()  # 梯度初始化 需要要在每个batch计算之前清除梯度
                sentences, tags = zip(*batch)  # 从batch中拿到句子，目标标签和原始长度
                lengths = sentences[1]  # 先取出原始长度项给length
                sentences = sentences[0]  # 然后把正式摘要数据项给sentences
                tags = tags[0]  # 取出关键词项给tags
                loss = self.model.neg_log_likelihood(sentences, tags, lengths)  # 损失计算
                epoch_loss += loss
                print("""epoch [{}]  {}/{} \tloss {:.2f}""".format(epoch, index, self.total_size,
                                                                   loss.cpu().tolist()[0]))
                # print(sentences.tolist())
                # 评估当前batch效果
                self.evaluate()
                # 反向传播优化
                loss.backward()
                optimizer.step()
                # 存储当前网络参数
            print("=======================================epoch_loss:", epoch_loss)
            torch.save(self.model.state_dict(), self.model_path + 'train500_sep_taged_batch128.pkl')

    def evaluate(self):
        sentences, labels = zip(*self.dev_batch.__next__())  # 取得一个batch的训练数据
        lengths = sentences[1]  # 先取出原始长度项给length
        sentences = sentences[0]  # 然后把正式摘要数据项给sentences
        # print(sentences.tolist())
        labels = labels[0]  # 取出关键词项给tags
        with torch.no_grad():
            _, paths = self.model(sentences, lengths)  # 代入模型计算得到输出为paths
        print(sentences.tolist())
        print(paths)
        print(labels.tolist())
        f1_score(labels, paths, self.model.tag_map)  # 代入计算函数

    def PRF1(self):
        for i in range(10):
            # sentences, tags = zip(*batch)
            sentences, labels = zip(*self.dev_batch.__next__())  # 取得一个batch的训练数据
            lengths = sentences[1]  # 先取出原始长度项给length
            sentences = sentences[0]  # 然后把正式摘要数据项给sentences
            labels = labels[0]  # 取出关键词项给tags
            print(sentences[1])
            # print(labels[1])
            with torch.no_grad():
                _, paths = self.model(sentences,lengths)  # 代入模型计算得到输出为paths
            print(paths[1])
            print(labels[1])
            f1_score1(labels, paths, self.model.tag_map)  # 代入计算函数


if __name__ == "__main__":
    keywords = keywords_extraction("train")
    keywords.train()
    keywords.PRF1()
    keywords = keywords_extraction("test")
    keywords.PRF1()
