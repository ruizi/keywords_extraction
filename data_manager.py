# -*- coding: utf-8 -*-
'''
@Author  : cai rui
@Date    : 2020/4/12 11:24 下午
'''
import copy
import pickle as cPickle
import json


class DataManager():
    def __init__(self, max_length=100, batch_size=20, data_type='train'):
        self.index = 0
        self.input_size = 0
        self.batch_size = batch_size  # 初始化batch大小
        self.max_length = max_length  # 初始化batch中最长序列长度
        self.data_type = data_type  # 这是一个控制器，用来选择是提取训练数据/测试数据
        self.data = []
        self.batch_data = []
        self.vocab = {"unk": 0, "bat_pad": 1}
        self.tag_map = {"B": 0, "I": 1, "O": 2, "START": 3, "STOP": 4}  # 这里改为 B I O Start Stop |OK

        if data_type == "train":
            self.data_path = "Data/kp20k_train20k_sep_taged.json"  # 制定数据路径
        elif data_type == "dev":
            self.data_path = "Data/kp20k_valid2k_sep_taged.json"
            self.load_data_map()  # 验证/测试集就先直接进相应函数
        elif data_type == "test":
            self.data_path = "data/kp20k_valid2k_sep_taged.json"
            self.load_data_map()  # 测试集就先直接进相应函数，载入训练好的模型参数

        self.load_data()  # 训练集载入，load_data做的工作是读入word-tag数据，word组成sentence存好，tag组成target存好，同时构建好词表
        self.prepare_batch()  # 准备batch

    def load_data_map(self):  # 从存储好的模型中载入词表和tag表
        with open("models/data.pkl", "rb") as f:
            self.data_map = cPickle.load(f)
            self.vocab = self.data_map.get("vocab", {})
            self.tag_map = self.data_map.get("tag_map", {})
            self.tags = self.data_map.keys()

    def load_data(self):  # 数据集载入->清理->存入data
        # load data
        # add vocab
        # covert to one-hot  使用的是one-hot编码
        # 数据读入块
        training_data1 = []

        with open(self.data_path) as fp:  # 从在init中指定的路径读取数据
            data = json.load(fp)
            for paper in data:
                abstract = paper['abstract']
                tags = paper['tags']
                for word in abstract:  # 给每一个不重复的词进行编码，比如‘HELLO WORLD’就是{'HELLO':0,'WORLD:1'}
                    if word not in self.vocab and self.data_type == "train":
                        self.vocab[word] = max(self.vocab.values()) + 1  # 构建词表
                sentence = [self.vocab.get(word, 0) for word in abstract]  # 使用one_hot编码重新组合成句子了
                target = [self.tag_map.get(tag, 0) for tag in tags]  # 把正确序列标注的one_hot编码放入target
                self.data.append([sentence, target])  # 一个句子到了最后一行的时候把sentence和tag组合放入data

        json_vocab = json.dumps(self.vocab)
        with open('Data/word_vocabulary/vocab.json', 'w') as json_file:
            json_file.write(json_vocab)

        self.input_size = len(self.vocab.values())  # 输入的大小维度是词表的长度
        print("{} data: {}".format(self.data_type, len(self.data)))
        print("vocab size: {}".format(self.input_size))
        print("unique tag: {}".format(len(self.tag_map.values())))
        print("-" * 50)

    def convert_tag(self, data):
        _, tags = data
        converted_tags = []
        for _, tag in enumerate(tags[:-1]):
            if tag not in self.tag_map and self.data_type == "train":
                self.tag_map[tag] = len(self.tag_map.keys())
            converted_tags.append(self.tag_map.get(tag, 0))
        converted_tags.append(0)
        data[1] = converted_tags
        assert len(converted_tags) == len(tags), "convert error, the list dosen't match!"
        return data

    def prepare_batch(self):
        '''
            prepare data for batch
        '''
        index = 0  # index记录当前批batch的起点位置
        while True:
            if index + self.batch_size >= len(self.data):  # 如果到了最后一个batch数据不足一整个batch_size
                pad_data = self.pad_data(self.data[-self.batch_size:])  # 等于是差动，如果数据不能填满窗口，就把窗口前移动，重复一部分数据
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index:index + self.batch_size])  # 正常情况下直接取batch_size大小数据就好
                index += self.batch_size
                self.batch_data.append(pad_data)

    def pad_data(self, data):  # 传入batch_size条数据，该函数用来长短对齐
        c_data = copy.deepcopy(data)
        max_length = max([len(i[0]) for i in c_data])  # 得到batch句子的最大长度
        # print("max_length:")
        # print(max_length)
        for i in c_data:
            i.append(len(i[0]))  # 加入一个sentence的长度在末尾
            i[0] = i[0] + (max_length - len(i[0])) * [1]  # 这是在末尾补占位符，来使得大家长短一样
            i[1] = i[1] + (max_length - len(i[1])) * [2]
            # i[0] = torch.tensor(i[0])
            # i[1] = torch.tensor(i[1])
        return c_data

    def iteration(self):  # 循环输出batch
        idx = 0
        while True:
            yield self.batch_data[idx]  # 返回第idx个batch
            idx += 1
            if idx > len(self.batch_data) - 1:
                idx = 0

    def get_batch(self):  # 返回batch
        for data in self.batch_data:
            yield data
