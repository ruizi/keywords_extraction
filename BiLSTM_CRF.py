# -*- coding: utf-8 -*-
'''
@Author  : cai rui
@Date    : 2020/4/21 4:14 下午
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import json
import os

START_TAG = "START"
STOP_TAG = "STOP"


# Compute log sum exp in a numerically stable way for the forward algorithm
# 以数值稳定的方式为前向算法计算对数和exp

def log_sum_exp1(vec):
    max_score = torch.max(vec, 0)[0].unsqueeze(0)
    max_score_broadcast = max_score.expand(vec.size(1), vec.size(1))
    result = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 0)).unsqueeze(0)
    return result.squeeze(1)
# Compute log sum exp in a numerically stable way for the forward algorithm

# 以数值稳定的方式为前向算法计算对数和exp
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # ??
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    # return the argmax as a python int 返回最大值所在的下标位置 # 得到最大的值的索引
    _, idx = torch.max(vec, 1)  # 返回每行中最大的元素和最大元素的索引
    return idx.item()


# 模型构建
class BiLSTM_CRF(nn.Module):
    """
        init model:
        the parameters：
        vocab_size：语料的字典的长度
        tag_map：标签与对应编号的字典
        embedding_dim：标签的数量
        hidden_dim：BiLSTM的隐藏层的神经元数量
        batch_size: batch大小
        dropout
    """

    def __init__(self, vocab_size, tag_map, embedding_dim, hidden_dim, batch_size=20, dropout=1.0):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  # 使用传入的词向量维度来初始化神经网络输入层的输入向量维度
        self.hidden_dim = hidden_dim  # BiLSTM的隐藏层的神经元数量
        self.vocal_size = vocab_size  # 语料的字典的长度
        self.tag_map = tag_map  # BIO Start stop标签字典
        self.tag_size = len(tag_map)  # 标签的类型数

        self.batch_size = batch_size
        self.dropout = dropout

        # 转移矩阵，transitions[i][j]表示从label j转移到label i的概率,虽然是随机生成的但是后面会迭代更新
        # 这里特别要注意的是transitions[i]表示的是其他标签转到标签i的概率 transitioning *to* i *from* j.
        self.transitions = nn.Parameter(torch.randn(self.tag_size, self.tag_size))  # 随机初始化一个参数出来
        self.transitions = nn.Parameter(
            torch.randn(self.tag_size, self.tag_size)
        )
        # 以下两个语句强制执行了这样的约束条件：我们永远不会转移到开始标签，也永远不会从停止标签转移
        self.transitions.data[tag_map[START_TAG], :] = -100.  # 从任何标签转移到START_TAG不可能
        self.transitions.data[:, tag_map[STOP_TAG]] = -100.  # 从STOP_TAG转移到任何标签不可能

        # 输出为一个mini-batch*words_num*embedding_dim的矩阵
        # vocab_size表示一共有多少词，embedding_dim表示想为每个词创建一个多少维的向量来表示
        self.word_embeds = nn.Embedding(vocab_size, self.embedding_dim)
        # 初始化创建LSTM(每个词的特征数量，隐藏层的特征数量，循环层的数量指堆叠的层数，是否是BiLSTM)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        self.hidden = self.init_hidden()

        # 将LSTM的输出映射到标签空间 Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)  # 就等于在神经网络输出层做一次降维，把输出降到标签个数的维度

        # 初始化了 h0 c0
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # （num_layers*num_directions,minibatch_size,hidden_dim）
        # 实际上初始化的h0和c0
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentences):  # 仅仅是BiLSTM的输出没有到CRF层
        '''
        LSTM的数据输入是以这样的形式输入的：（序列长度，batch，输入大小），下面的是LSTM的文档中定义的使用方法
        input_size指输入数据的维度
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        '''

        self.hidden = self.init_hidden()  # lstm使用前先初始化h0 c0
        length = sentences.shape[1]  # 通过shape可以拿到该batch中sentence的统一长度，因为在data_manager中使用了pad填充到了统一长度
        embeddings = self.word_embeds(sentences).view(self.batch_size, length, self.embedding_dim)  # 词嵌入，随机初始化表示
        # batch_size*length*embedding_dim    就是batch_size等于句子数，length是句子长度，embedding_dim句子中每个词的维度
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)  # 把准备好的h0，c0和embedding后的输入句放入lstm运算
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        logits = self.hidden2tag(lstm_out)  # 维度降到tag_size
        return logits  # 返回每个词在lstm输出后的标签概率矩阵

    # def _score_sentence(self, feats, tags):  # 求Loss function的第二项
    #     # Gives the score of a provided tag sequence
    #     # 这与上面的def _forward_alg(self, feats)共同之处在于：
    #     # 两者都是用的随机转移矩阵算的score，不同地方在于，上面那个函数算了一个最大可能路径，
    #     # 但实际上可能不是真实的各个标签转移的值 例如：真实标签是N V V,但是因为transitions是随机的，
    #     # 所以上面的函数得到其实是N N N这样，两者之间的score就有了差距。而后来的反向传播，就能够更新transitions，
    #     # 使得转移矩阵逼近真实的“转移矩阵”得到gold_seq tag的score 即根据真实的label 来计算一个score，
    #     # 但是因为转移矩阵是随机生成的，故算出来的score不是最理想的值
    #     score = torch.zeros(1)  # 初始化分数为0
    #     # 将START_TAG的标签３拼接到tag序列最前面  B  I  O  START_TAG STOP_TAG
    #     tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
    #     for i, feat in enumerate(feats):  # 循环各个词
    #         # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
    #         # feat[tags[i+1]], feat是step i 的输出结果，有５个值，
    #         # 对应B, I, O, START_TAG, END_TAG, 取对应标签的值
    #         # transition【j,i】 就是从i ->j 的转移概率值
    #         score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
    #     score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
    #     return score

    # def _viterbi_decode(self, feats):
    #     # 预测序列的得分，维特比解码，输出得分与路径值
    #     backpointers = []
    #     # Initialize the viterbi variables in log space
    #     init_vvars = torch.full((1, self.tag_size), -10000.)  # 这就保证了一定是从START到其他标签
    #     init_vvars[0][self.tag_map[START_TAG]] = 0  # -10000,-10000,-10000,0,-10000
    #
    #     # forward_var at step i holds the viterbi variables for step i-1
    #     forward_var = init_vvars
    #     for feat in feats:
    #         bptrs_t = []  # holds the backpointers for this step 记录路径
    #         viterbivars_t = []  # holds the viterbi variables for this step
    #
    #         for next_tag in range(self.tag_size):
    #             # 其他标签（B,I,O,Start,End）到标签next_tag的概率
    #             # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
    #             # plus the score of transitioning from tag i to next_tag.
    #             # We don't include the emission scores here because the max
    #             # does not depend on them (we add them in below)
    #             next_tag_var = forward_var + self.transitions[next_tag]  # forward_var保存的是之前的最优路径的值
    #             best_tag_id = argmax(next_tag_var)  # 返回最大值对应的那个tag
    #             bptrs_t.append(best_tag_id)  # 加入路径列表
    #             viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
    #         # Now add in the emission scores, and assign forward_var to the set
    #         # of viterbi variables we just computed
    #         # viterbivars_t是记录了从step0到step(i-1)时5个序列中每个序列的最大score
    #         forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
    #         backpointers.append(bptrs_t)  # bptrs_t有５个元素
    #
    #     # 其他标签到STOP_TAG的转移概率
    #     terminal_var = forward_var + self.transitions[self.tag_map[STOP_TAG]]
    #     best_tag_id = argmax(terminal_var)
    #     path_score = terminal_var[0][best_tag_id]
    #
    #     # Follow the back pointers to decode the best path. 通过记录的路径解码最优路径
    #     best_path = [best_tag_id]
    #     for bptrs_t in reversed(backpointers):
    #         best_tag_id = bptrs_t[best_tag_id]
    #         best_path.append(best_tag_id)
    #     # Pop off the start tag (we dont want to return that to the caller)
    #     start = best_path.pop()
    #     assert start == self.tag_map[START_TAG]  # Sanity check
    #     best_path.reverse()  # 把从后向前的路径正过来
    #     return path_score, best_path

    def _forward_alg(self, feats):
        # 预测序列的得分，就是Loss的右边第一项
        # feats表示发射矩阵(emit score)，实际上就是LSTM的输出，意思是经过LSTM的sentence的每个word对应于每个label的得分
        init_alphas = torch.full((1, self.tag_size), -10000.)  # 用-10000.来填充一个形状为[1,tagset_size]的tensor
        # START_TAG has all of the score.
        # 因为start tag是4，所以tensor([[-10000., -10000., -10000., 0., -10000.]])， B,I,O,Start,Stop
        # 将start的值为零，表示开始进行网络的传播，
        init_alphas[0][self.tag_map[START_TAG]] = 0

        # 包装到一个变量里面以便自动反向传播
        forward_var = init_alphas  # 初始状态的forward_var，随着step t变化

        # 遍历句子，迭代feats的行数次，就是一个词一个词遍历 每个feat是一个词嵌入经过lstm和降维后的标签概率矩阵
        for feat in feats:
            alphas_t = []  # 当前时间步的正向tensor
            for next_tag in range(self.tag_size):  # 行遍历计算转移分数,   ---->循环计算每个tag，tagset就是B，I，O，ST，SP
                # broadcast the emission score: it is the same regardless of the previous tag
                # LSTM的生成矩阵是emit_score，维度为1*5
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tag_size)  # 取出当前词对于当前tag对象的概率后扩充为5个
                # the ith entry of trans_score is the score of transitioning to next_tag from i
                # trans_score的第i个条目是从i过渡到next_tag的分数 ，取转移矩阵的对应tag的行，该行中每列表示的是该列对应的tag转移到当前tag的概率
                trans_score = self.transitions[next_tag].view(1, -1)  # view(1, -1)转化为一行维度是1*5
                # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
                # 第一次迭代时理解：trans_score是所有其他标签到Ｂ标签的概率
                # emit_scores是由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score  # 这里就是公式中的简易计算方法，通过前一个tag的计算参数推导后一个
                # The forward variable for this tag is log-sum-exp of all the scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            # 此时的alphas_t 是一个长度为5，例如<class 'list'>:
            # [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
            forward_var = torch.cat(alphas_t).view(1, -1)  # 计算好一个tag的前向参数概率后把该值融入前向变量

        # 最后只将最后一个单词的forward_var与转移 stop tag的概率相加
        # tensor([[   21.1036,    18.8673,    20.7906, -9982.2734, -9980.3135]])
        terminal_var = forward_var + self.transitions[self.tag_map[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)  # alpha是一个0维的tensor
        return alpha

    # viterbi解码faster
    def _viterbi_decode_faster(self, logits):
        backpointers = []
        trellis = torch.zeros(logits.size())
        backpointers = torch.zeros(logits.size(), dtype=torch.long)

        trellis[0] = logits[0]
        for t in range(1, len(logits)):
            v = trellis[t - 1].unsqueeze(1).expand_as(self.transitions) + self.transitions
            trellis[t] = logits[t] + torch.max(v, 0)[0]
            backpointers[t] = torch.max(v, 0)[1]
        viterbi = [torch.max(trellis[-1], -1)[1].cpu().tolist()]
        backpointers = backpointers.numpy()
        for bp in reversed(backpointers[1:]):
            viterbi.append(bp[viterbi[-1]])
        viterbi.reverse()

        viterbi_score = torch.max(trellis[-1], 0)[0].cpu().tolist()
        return viterbi_score, viterbi

    def _viterbi_decode(self, feats):
        # 预测序列的得分，维特比解码，输出得分与路径值
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tag_size), -100.)  # 这就保证了一定是从START到其他标签
        init_vvars[0][self.tag_map[START_TAG]] = 0  # -100,-100,-100,0,-100

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step 记录路径
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tag_size):
                # 其他标签（B,I,O,Start,End）到标签next_tag的概率
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step,
                # plus the score of transitioning from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]  # forward_var保存的是之前的最优路径的值
                best_tag_id = argmax(next_tag_var)  # 返回最大值对应的那个tag
                bptrs_t.append(best_tag_id)  # 加入路径列表
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # viterbivars_t是记录了从step0到step(i-1)时5个序列中每个序列的最大score
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)  # bptrs_t有５个元素

        # 其他标签到STOP_TAG的转移概率
        terminal_var = forward_var + self.transitions[self.tag_map[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path. 通过记录的路径解码最优路径
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_map[START_TAG]  # Sanity check
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path

    def real_path_score(self, logit, label):  # 这里输入为batch中的一个句子和对应的正确标签 # loss的后半部分S(X,y)的结果 gold_score
        '''
        caculate real path score
        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * len_sent]

        Score = Emission_Score + Transition_Score
        Emission_Score = logits(0, label[START]) + logits(1, label[1]) + ... + logits(n, label[STOP])
        Transition_Score = Trans(label[START], label[1]) + Trans(label[1], label[2]) + ... + Trans(label[n-1], label[STOP])
        '''
        score = torch.zeros(1)  # 初始化得分为0
        label = torch.cat([torch.tensor([self.tag_map[START_TAG]], dtype=torch.long), label])  # 在label的开头加入start_tag
        for index, lstm_output in enumerate(logit):  # lstm_output是bilstm对一个句子中词的输出结果，这里是在遍历句子，也就是遍历每个时间步
            emission_score = lstm_output[label[index + 1]]
            # label[index+1]是该word的正确标签，这里就是取每个词的输出标签概率矩阵中正确的那个值
            # print("word:%s emission_score:%s", index, str(emission_score))
            transition_score = self.transitions[label[index + 1], label[index]]
            # print(label[index], label[index + 1])
            # print(transition_score)
            # 转移矩阵是上一个标签到本标签的转移概率 label是正确的标签序列
            score += emission_score + transition_score
            # print(score)
        score += self.transitions[self.tag_map[STOP_TAG], label[-1]]  # 加上最后的终止标签，只有转移概率，发射矩阵不包括stop tag
        # print(score)
        return score

    def total_score(self, logits):  # loss右边第一项
        """
        caculate total score

        :params logits -> [len_sent * tag_size]
        :params label  -> [1 * tag_size]

        SCORE = log(e^S1 + e^S2 + ... + e^SN)
        """
        obs = []
        previous = torch.full((1, self.tag_size), 0)
        for index in range(len(logits)):
            previous = previous.expand(self.tag_size, self.tag_size).t()  # 扩大后转置
            obs = logits[index].view(1, -1).expand(self.tag_size, self.tag_size)
            scores = previous + obs + self.transitions
            previous = log_sum_exp1(scores)
        previous = previous + self.transitions[self.tag_map[STOP_TAG], :]
        # caculate total_scores
        total_scores = log_sum_exp1(previous.t())[0]
        return total_scores

    def neg_log_likelihood(self, sentences, tags, length):  # loss计算
        self.batch_size = sentences.size(0)  # 通过输入的sentences的tensor的大小取得batch大小

        logits = self._get_lstm_features(sentences)  # 经过LSTM+Linear后的输出作为CRF的输入

        real_path_score = torch.zeros(1)
        total_score = torch.zeros(1)

        # print("transitions")
        # print(self.transitions)

        for logit, tag, leng in zip(logits, tags, length):
            logit = logit[:leng]  # 把输出和原tag序列都截取回本身的长度，即去掉pad，为了batch输入统一了长度，使用pad补足，现在去掉
            tag = tag[:leng]
            real_path_score += self.real_path_score(logit, tag)  # 这里得到的是loss的后半部分S(X,y)的结果，是根据真实label计算得到的输出序列的分数
            total_score += self._forward_alg(logit)  # 当前最大得分
        # print("total score ", total_score)
        # print("real score ", real_path_score)
        return total_score - real_path_score  # 返回当前loss

    def forward(self, sentences, lengths=None):  # sentences是已经编码的句子 网络自动调用，重载函数
        # # Get the emission scores from the BiLSTM
        # lstm_feats = self._get_lstm_features(sentence)  # 从bilstm得到发射矩阵
        # score, tag_seq = self._viterbi_decode(lstm_feats)  # Find the best path, given the features.
        # return score, tag_seq

        """
           :params sentences sentences to predict
           :params lengths represent the ture length of sentence, the default is sentences.size(-1)
        """
        sentences = torch.tensor(sentences, dtype=torch.long)
        if not lengths:
            lengths = [i.size(-1) for i in sentences]
        self.batch_size = sentences.size(0)  # 取得batch大小
        logits = self._get_lstm_features(sentences)  # 代入lstm得到lstm层的输出
        scores = []
        paths = []
        for logit, leng in zip(logits, lengths):
            logit = logit[:leng]  # 去掉pad
            score, path = self._viterbi_decode_faster(logit)  # 使用viterbi算法解码得到最佳路径
            scores.append(score)
            paths.append(path)
        return scores, paths
