from clean_and_tag import abstract_keyword
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import json
import os
# Run training
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5  # 由于标签一共有B\I\O\START\STOP 5个，所以embedding_dim为5
HIDDEN_DIM = 4  # 这其实是BiLSTM的隐藏层的特征数量，因为是双向所以是2倍，单向为2
tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}  # tag的字典，每个tag对应一个编码
model1 = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)  # 模型
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)  # 优化器，采用随机梯度下降
checkpoint = torch.load('models/keywords_model.pth')
model1.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
# loss = checkpoint['loss']
model1.eval()

with torch.no_grad():
    precheck_sent = prepare_sequence(training_data1[0][0], word_to_ix)
    print(model(precheck_sent))
    print(model1(precheck_sent))