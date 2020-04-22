import json
import os
import random
import re
from clean_and_tag import abstract_keyword, abstract_keyword1, abstract_keyword2


# 该函数用来从原始kp20k数据集中提取摘要和关键词
def data_read_and_clean():
    examples = []
    trg_count = 0
    valid_trg_count = 0
    data = []
    for line_id, line in enumerate(open('Data/kp20k_train20k_test.json', 'r')):
        paper = {}
        print("Processing %d" % line_id)
        print(line)
        json_dict = json.loads(line)
        # may add more fields in the future, say dataset_name, keyword-specific features
        if 'id' in json_dict:
            id = json_dict['id']
        else:
            id = str(line_id)
        paper['id'] = str(line_id)
        paper['title'] = json_dict['title']
        paper['abstract'] = json_dict['abstract']
        keywords = json_dict['keywords']
        # print(keywords)
        # process strings
        # keywords may be a string concatenated by ';', make sure the output is a list of strings
        # if isinstance(keywords, str):
        #     keywords = keywords.split(';')
        #     paper['keywords'] = keywords
        # 在train中是直接为list不用手动切割
        paper['keywords'] = keywords
        # remove all the abbreviations/acronyms in parentheses in keyphrases
        # keywords = [re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', kw) for kw in keywords]
        # print(keywords)
        data.append(paper)
    with open('Data/kp20k_train20k.json', 'w') as fp:
        json.dump(data, fp=fp)


# data_short1.json / kp20k_valid500.json / Data/kp20k_valid2k.json /kp20k_train20k.json
def read_test():
    with open('Data/kp20k_valid500.json', 'r') as fp:
        data = json.load(fp)
    return data


def word2tag(data):
    data_tags = []
    index = 0
    for paper in data:
        print(paper)
        paper_tag = {}
        keywords = paper['keywords']
        abstract = paper['abstract']

        print("processing %s/20000" % index)

        abstract_clean, BIOtag = abstract_keyword1(abstract, keywords)
        if len(abstract_clean) == 0 or len(BIOtag) == 0:
            continue
        paper_tag['id'] = index
        paper_tag['abstract'] = abstract_clean
        paper_tag['tags'] = BIOtag
        data_tags.append(paper_tag)
        index += 1
    with open('Data/kp20k_valid500_short_taged.json', 'w') as fp:
        json.dump(data_tags, fp=fp)


def sentence_short2tag(data):
    data_tags = []
    index = 0
    for paper in data:
        print(paper)
        paper_tag = {}
        keywords = paper['keywords']
        abstract = paper['abstract']

        print("processing %s/20000" % index)

        short_abstracts_sentence, short_abstracts_tag = abstract_keyword2(abstract, keywords)
        for short_sentence, short_tag in zip(short_abstracts_sentence, short_abstracts_tag):
            if len(short_sentence) == 0 or len(short_tag) == 0:
                continue
            paper_tag['id'] = index
            paper_tag['abstract'] = short_sentence
            paper_tag['tags'] = short_tag
            data_tags.append(paper_tag)
        index = index + 1

    with open('Data/kp20k_valid500_sep_taged.json', 'w') as fp:
        json.dump(data_tags, fp=fp)


# 1. 先调用清理函数得到规范化的json格式
# data_read_and_clean()
# 2. 把上一步写入的文件读入，存在data
data = read_test()
print(len(data))
# 3. 把标准化的json中的keywords和abstract 分词，词性还原，匹配打上tag后返回写入文件，到这里就完成bilstm-crf的输入
word2tag(data)
# sentence_short2tag(data)
