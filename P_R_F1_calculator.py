# -*- coding: utf-8 -*-
'''
@Author  : cai rui
@Date    : 2020/4/20 12:09 上午
'''


# 使用tag序列比对，不还原为原串

def get_tags(path, tag_map):
    begin_tag = tag_map.get("B")
    mid_tag = tag_map.get("I")
    o_tag = tag_map.get("O")
    begin = -1
    end = 0
    tags = []
    last_tag = 0
    index = 0
    for tag in path:
        if tag == begin_tag and index == 0:
            begin = 0
        elif tag == begin_tag:
            begin = index
        elif tag == o_tag and last_tag in [mid_tag, begin_tag] and begin > -1:
            end = index
            tags.append([begin, end - 1])
        elif tag == o_tag:
            begin = -1
        last_tag = tag
        index += 1
    return tags


# def f1_score(tar_path, pre_path, tag_map):  # tag是类型标签
#     origin = 0.
#     found = 0.
#     right = 0.
#
#     for fetch in zip(tar_path, pre_path):
#         tar, pre = fetch
#         # print("tar:" + str(tar))
#         # print("pre:" + str(pre))
#         tar_tags = get_tags(tar, tag_map)
#         pre_tags = get_tags(pre, tag_map)
#         # print("tar_tags:" + str(tar_tags))
#         # print("pre_tags:" + str(pre_tags))
#         origin += len(tar_tags)
#         found += len(pre_tags)
#
#         for p_tag in pre_tags:
#             if p_tag in tar_tags:
#                 right += 1
#
#     recall = 0. if origin == 0 else (right / origin)
#     precision = 0. if found == 0 else (right / found)
#     f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
#     print("recall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(recall, precision, f1))
#     return recall, precision, f1


def f1_score(tar_path, pre_path, tag_map):  # tag是类型标签
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        tar_tags = get_tags(tar, tag_map)
        pre_tags = get_tags(pre, tag_map)
        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1

    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    print("recall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(recall, precision, f1))
    return recall, precision, f1


def f1_score1(tar_path, pre_path, tag_map):  # tag是类型标签
    origin = 0.
    found = 0.
    right = 0.
    for fetch in zip(tar_path, pre_path):
        tar, pre = fetch
        tar_tags = get_tags(tar, tag_map)
        pre_tags = get_tags(pre, tag_map)
        # print(tar)
        # print("tar_tags:" + str(tar_tags))
        # print(pre)
        # print("pre_tags:" + str(pre_tags))
        origin += len(tar_tags)
        found += len(pre_tags)

        for p_tag in pre_tags:
            if p_tag in tar_tags:
                right += 1
        # print("right:", right)
    print("该batch中关键词个数：", origin)
    print("模型输出的关键词个数：", found)
    print("模型命中的关键词个数：", right)
    recall = 0. if origin == 0 else (right / origin)
    precision = 0. if found == 0 else (right / found)
    f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
    print("recall {:.2f}\tprecision {:.2f}\tf1 {:.2f}".format(recall, precision, f1))
    return recall, precision, f1
