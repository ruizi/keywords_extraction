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
                "embedding_size": 100,
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
        self.tags = config.get("tags")
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
            dev_manager = DataManager(batch_size=20, data_type="dev")
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
            # self.restore_model()

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
            self.restore_model()

        else:
            pass

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path + "params.pkl"))
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
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, weight_decay=1e-4)  # 定义优化器，采用随机梯度下降
        # optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(ner_model.parameters(), lr=0.01)
        for epoch in range(100):
            index = 0
            for batch in self.train_manager.get_batch():  # 从data_manager中导入配置
                index += 1  # 训练位置
                self.model.zero_grad()  # 梯度初始化 需要要在每个batch计算之前清除梯度

                sentences, tags, length = zip(*batch)  # 从batch中拿到句子，目标标签和原始长度

                sentences_tensor = torch.tensor(sentences, dtype=torch.long)  # 三个list都转化为tensor
                tags_tensor = torch.tensor(tags, dtype=torch.long)
                length_tensor = torch.tensor(length, dtype=torch.long)

                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, length_tensor)  # 损失计算

                # progress = ("█" * int(index * 25 / self.total_size)).ljust(25)  # 进度条
                print("""epoch [{}]  {}/{}\n\tloss {:.2f}""".format(
                    epoch, index, self.total_size, loss.cpu().tolist()[0]
                ))
                # 评估当前batch效果
                self.evaluate()
                # 反向传播优化
                loss.backward()
                optimizer.step()
                # 存储当前网络参数
            torch.save(self.model.state_dict(), self.model_path + 'params.pkl')

    def evaluate(self):
        sentences, labels, length = zip(*self.dev_batch.__next__())  # 取得一个batch的训练数据
        # print(sentences)
        # print(labels)
        _, paths = self.model(sentences)  # 代入模型计算得到输出为paths
        # print(len(paths[1]))
        # print(paths[0])
        # print(len(labels[0]))
        # print(labels[0])
        print("\tevaluate_this_batch")
        f1_score(labels, paths, self.model.tag_map)  # 代入计算函数

    def PRF1(self):
        sentences, labels, length = zip(*self.dev_batch.__next__())  # 取得一个batch的训练数据
        print(sentences[6])
        print(labels[6])
        with torch.no_grad():
            _, paths = self.model(sentences)  # 代入模型计算得到输出为paths
        print(paths[6])
        print(labels[6])
        f1_score1(labels, paths, self.model.tag_map)  # 代入计算函数


# def train():
#     # Run training
#     # START_TAG = "<START>"
#     # STOP_TAG = "<STOP>"
#     # EMBEDDING_DIM = 5  # 由于标签一共有B\I\O\START\STOP 5个，所以embedding_dim为5
#     # HIDDEN_DIM = 4  # 这其实是BiLSTM的隐藏层的特征数量，因为是双向所以是2倍，单向为2
#
#     # # 训练数据
#     # training_data = [(
#     #     "the wall street journal reported today that apple corporation made money".split(),
#     #     # ['the','wall','street','journal',...]
#     #     "B I I I O O O B I O O".split()  # ['B','I','I',...]
#     # ), (
#     #     "georgia tech is a university in georgia".split(),
#     #     "B I O O O O B".split()
#     # )]
#     #
#     # 数据读入块
#
#     training_data1 = []
#     with open('Data/ori_test_used/data_short_tag.json', 'r') as fp:
#         data = json.load(fp)
#         print(data)
#         for paper in data:
#             temp_tup = (paper['abstract'], paper['tags'])
#             training_data1.append(temp_tup)
#         print(training_data1)
#         # print(training_data)
#
#     # 给每一个不重复的词进行编码，比如‘HELLO WORLD’就是{'HELLO':0,'WORLD:1'}
#     word_to_ix = {}  # 训练语料的字典，语料中的每一个字对应的编码(index)
#     for sentence, tags in training_data1:
#         for word in sentence:
#             if word not in word_to_ix:
#                 word_to_ix[word] = len(word_to_ix)
#
#     tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}  # tag的字典，每个tag对应一个编码
#     model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)  # 模型初始化定义
#     optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)  # 定义优化器，采用随机梯度下降
#     print(len(word_to_ix))
#     # Check predictions before training 初始化预测
#     with torch.no_grad():
#         precheck_sent = prepare_sequence(training_data1[0][0], word_to_ix)
#         precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data1[0][1]], dtype=torch.long)
#         print(model(precheck_sent))  # (tensor(2.6907), [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1])
#
#     # Make sure prepare_sequence from earlier in the LSTM section is loaded
#     for epoch in range(100):  # 循环次数可以自己设定
#         index = 0
#         for sentence, tags in training_data1:
#             index += 1
#             # Step 1. Remember that Pytorch accumulates gradients.
#             # We need to clear them out before each instance(
#             model.zero_grad()
#             # print(tags)
#             # Step 2. Get our inputs ready for the network, that is,
#             # turn them into Tensors of word indices.
#             sentence_in = prepare_sequence(sentence, word_to_ix)
#             targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
#
#             # Step 3. Run our forward pass.
#             loss = model.neg_log_likelihood(sentence_in, targets)
#
#             # Step 4. Compute the loss, gradients, and update the parameters by
#             # calling optimizer.step()
#             loss.backward()
#             optimizer.step()
#         print("epoch:%s", epoch)
#     # Check predictions after training
#     torch.save({"epoch": 30,  # 一共训练的epoch
#                 "model_state_dict": model.state_dict(),  # 保存模型参数×××××这里埋个坑××××
#                 "optimizer_state_dict": optimizer.state_dict()  # 优化器好像也在保存，这样可以继续加载模型进行训练
#                 }, 'models/keywords_model.pth')
#
#     with torch.no_grad():
#         precheck_sent = prepare_sequence(training_data1[0][0], word_to_ix)
#         print(model(precheck_sent))
#         # print(model1(precheck_sent))


# def load_model():
#     # device = torch.device("cuda")
#     # model = TheModelClass(*args, **kwargs)
#     # model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Choose whatever GPU device number you want
#     #
#     #
#     # model.to(device)# Make sure to call input = input.to(device) on any input tensors that you feed to the model
#     training_data1 = []
#     with open('Data/ori_test_used/data_short_tag.json', 'r') as fp:
#         data = json.load(fp)
#         print(data)
#         for paper in data:
#             temp_tup = (paper['abstract'], paper['tags'])
#             training_data1.append(temp_tup)
#         print(training_data1)
#         # print(training_data)
#
#     # 给每一个不重复的词进行编码，比如‘HELLO WORLD’就是{'HELLO':0,'WORLD:1'}
#     word_to_ix = {}  # 训练语料的字典，语料中的每一个字对应的编码(index)
#     for sentence, tags in training_data1:
#         for word in sentence:
#             if word not in word_to_ix:
#                 word_to_ix[word] = len(word_to_ix)
#
#     tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}  # tag的字典，每个tag对应一个编码
#     model1 = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)  # 模型
#     optimizer = optim.SGD(model1.parameters(), lr=0.01, weight_decay=1e-4)  # 优化器，采用随机梯度下降
#     checkpoint = torch.load('models/keywords_model.pth')
#     model1.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     # loss = checkpoint['loss']
#     model1.eval()
#     with torch.no_grad():
#         targets = []
#         predicts = []
#         for sentence, tags in training_data1:
#             one_hot_input = prepare_sequence(sentence, word_to_ix)
#             predict = model1(one_hot_input)[1]
#             target = ([tag_to_ix[t] for t in tags])
#             predicts.append(predict)
#             targets.append(target)
#             # print(model(precheck_sent))
#         f1_score(targets, predicts, tag_to_ix)  # 代入计算函数
#         print()


# train()
# load_model()

if __name__ == "__main__":
    keywords = keywords_extraction("train")
    keywords.train()
    keywords.PRF1()
