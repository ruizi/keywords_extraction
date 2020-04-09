import json
import os
import random
import re
from learn_use_temp import abstract_keyword


def data_read_and_clean():
    examples = []
    trg_count = 0
    valid_trg_count = 0
    data = []
    for line_id, line in enumerate(open('data_short.json', 'r')):
        paper = {}
        print("Processing %d" % line_id)
        print(line)
        json_dict = json.loads(line)
        # may add more fields in the future, say dataset_name, keyword-specific features
        if 'id' in json_dict:
            id = json_dict['id']
        else:
            id = str(line_id)
        paper['id'] = id
        paper['title'] = json_dict['title']
        paper['abstract'] = json_dict['abstract']
        keywords = json_dict['keywords']

        # process strings
        # keywords may be a string concatenated by ';', make sure the output is a list of strings
        if isinstance(keywords, str):
            keywords = keywords.split(';')
            paper['keywords'] = keywords

        # remove all the abbreviations/acronyms in parentheses in keyphrases
        # keywords = [re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', kw) for kw in keywords]
        # print(keywords)
        data.append(paper)
    with open('data_short1.json', 'w') as fp:
        json.dump(data, fp=fp)


def read_test():
    with open('data_short1.json', 'r') as fp:
        data = json.load(fp)
    return data


def word2tag(data):
    data_tags = []
    for paper in data:
        paper_tag = {}
        keywords = paper['keywords']
        abstract = paper['abstract']
        print(keywords)
        print(abstract)
        print("*" * 50)

        abstract_clean, BIOtag = abstract_keyword(abstract, keywords)
        paper_tag['abstract'] = abstract_clean
        paper_tag['tags'] = BIOtag
        data_tags.append(paper_tag)
        # for iter in abstract_list:
        #     if iter in keywords_list:
        #         BIOtags.append("B")
        #     else if iter ==

        # data_read_and_clean()
    with open('data_short_tag.json', 'w') as fp:
        json.dump(data_tags, fp=fp)


data = read_test()

print(len(data))
word2tag(data)
