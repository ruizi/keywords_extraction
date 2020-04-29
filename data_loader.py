# -*- coding: utf-8 -*-
'''
@Author  : cai rui
@Date    : 2020/4/27 3:27 下午
'''

# -*- coding: utf-8 -*-
'''
@Author  : cai rui
@Date    : 2020/4/12 11:24 下午
'''
import copy
import pickle as cPickle
import json
import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
import torch.utils.data as data


class MyData(data.Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]


def lens_sort_and_pad_for_sentences(raw_sentences):  # sentences的pad使用0
    raw_sentences.sort(key=lambda x: len(x), reverse=True)  # 长度排序
    raw_sentences_length = [len(sq) for sq in raw_sentences]  # 记录pad前的长度
    processed_sentences = rnn_utils.pad_sequence(raw_sentences, batch_first=True, padding_value=0)  # B*S*D
    return processed_sentences.unsqueeze(-1), raw_sentences_length


def lens_sort_and_pad_for_tags(raw_tags):  # tags的pad使用2 表示为BIO序列中的O
    raw_tags.sort(key=lambda x: len(x), reverse=True)  # 长度排序
    raw_tags_length = [len(sq) for sq in raw_tags]  # 记录pad前的长度
    processed_tags = rnn_utils.pad_sequence(raw_tags, batch_first=True, padding_value=2)  # B*S*D
    return processed_tags, raw_tags_length


class DataManager():
    def __init__(self, batch_size=20, data_type='train'):
        self.input_size = 0
        self.batch_size = batch_size  # 初始化batch大小
        self.data_type = data_type  # 这是一个控制器，用来选择是提取训练数据/测试数据
        self.sentences = []
        self.tags = []
        self.batch_data = []
        self.vocab = {"unk": 0, "bat_pad": 1}
        self.tag_map = {"B": 0, "I": 1, "O": 2, "START": 3, "STOP": 4}  # 这里改为 B I O Start Stop |OK

        if data_type == "train":
            self.data_path = "Data/kp20k_valid500_sep_taged.json"  # 制定数据路径
        elif data_type == "dev":
            self.data_path = "Data/kp20k_valid500_sep_taged.json"
            self.load_data_map()  # 验证/测试集就先直接进相应函数
        elif data_type == "test":
            self.data_path = "data/kp20k_valid500_sep_taged.json"
            self.load_data_map()  # 测试集就先直接进相应函数，载入训练好的模型参数

        self.load_data()  # 训练集载入，load_data做的工作是读入word-tag数据，word组成sentence存好，tag组成target存好，同时构建好词表
        self.prepare_batch()  # 准备batch

    def load_data_map(self):  # 从存储好的模型中载入词表和tag表
        with open("models/data.pkl", "rb") as f:
            self.data_map = cPickle.load(f)
            self.vocab = self.data_map.get("vocab", {})
            self.tag_map = self.data_map.get("tag_map", {})

    def load_data(self):  # 数据集载入->清理->存入data
        # 使用的是one-hot编码/构建词表
        with open(self.data_path) as fp:  # 从在init中指定的路径读取数据
            data = json.load(fp)
            for paper in data:
                abstract = paper['abstract']
                BIO_tags = paper['tags']
                for word in abstract:  # 给每一个不重复的词进行编码，比如‘HELLO WORLD’就是{'HELLO':0,'WORLD:1'}
                    if word not in self.vocab and self.data_type == "train":
                        self.vocab[word] = max(self.vocab.values()) + 1  # 构建词表
                sentence = [self.vocab.get(word, 0) for word in abstract]  # 使用one_hot编码重新组合成句子了
                target = [self.tag_map.get(BIO_tag, 0) for BIO_tag in BIO_tags]  # 把正确序列标注的one_hot编码放入target

                self.sentences.append(torch.tensor(sentence, dtype=torch.long))
                self.tags.append(torch.tensor(target, dtype=torch.long))

        json_vocab = json.dumps(self.vocab)
        with open('Data/word_vocabulary/vocab.json', 'w') as json_file:
            json_file.write(json_vocab)

        self.input_size = len(self.vocab.values())  # 输入的大小维度是词表的长度
        print("{} data: {}".format(self.data_type, len(self.sentences)))
        print("vocab size: {}".format(self.input_size))
        print("unique tag: {}".format(len(self.tag_map.values())))
        print("-" * 50)

    def prepare_batch(self):
        # 在DataLoader中把数据打包成batch，这里关闭了随机，同时在使用pad将每个句子补充到一样长前先进行了排序
        self.sentences_loader = DataLoader(dataset=self.sentences,
                                           batch_size=self.batch_size,
                                           shuffle=False, collate_fn=lens_sort_and_pad_for_sentences)
        self.tags_loader = DataLoader(dataset=self.tags,
                                      batch_size=self.batch_size,
                                      shuffle=False, collate_fn=lens_sort_and_pad_for_tags)
        # index = 0
        # for sentences, tags in zip(self.sentences_loader, self.tags_loader):
        #     print("第", index, "个batch的\n sentences的原始长度: ", sentences[1])
        #     print("sentences: ", sentences)
        #     print("tags:", tags)
        #     index += 1

    def iteration(self):  # 循环输出batch
        idx = 0
        while True:
            for dev_sentences, dev_tags in zip(self.sentences_loader, self.tags_loader):
                yield zip(dev_sentences, dev_tags)
            # yield self.batch_data[idx]  # 返回第idx个batch
            idx += 1
            if idx > len(self.sentences_loader) - 1:
                idx = 0


    def get_batch(self):  # 返回batch
        for sentences, tags in zip(self.sentences_loader, self.tags_loader):
            yield zip(sentences, tags)

        # for data in self.batch_data:
        #     yield data
