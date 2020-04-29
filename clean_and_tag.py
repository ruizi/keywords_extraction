import re

from nltk.tag import _pos_tag

from nltk import word_tokenize, pos_tag, PerceptronTagger
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))  # 它是没有对中文设计的，中文使用jieba


def pos_tag(tokens, tagset=None):
    tagger = PerceptronTagger()
    return _pos_tag(tokens, tagset, tagger, lang='eng')


# 获取单词的词性 使用词性来控制词性还原器还原到正确词源
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def multi_keys(word, keyword_long_list, i, s1):  # word是当前主循环取词，keyword_long_list是一个长关键词切分为子list后的集合，i为当前循环位置
    for maykey in keyword_long_list:  # 循环遍历长关键词集合中所有的子list
        if maykey[0] == word:  # 如果子list的第一个词匹配了，判断后面的所有是否能匹配上
            keyword_long_length = len(maykey)  # 得到关键词长度
            temp_list = s1[i:i + keyword_long_length]  # 在原句中截取出对应的长度
            if temp_list == maykey:  # 尝试匹配
                return keyword_long_length  # 匹配上返回对应长度，用来标注B，I和控制主循环跳跃
        else:
            continue

    return -1  # 没有找到就不是关键词，返回-1


def abstruct_pos_process(sentence):
    tokens = word_tokenize(sentence)  # 分词
    tagged_sent = pos_tag(tokens)  # 获取单词词性

    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原

    return lemmas_sent


def abstract_keyword(abstract, keywords):
    # s = "We investigate the problem of delay constrained maximal information collection for CSMA-based wireless sensor networks. We study how to allocate the maximal allowable transmission delay at each node, such that the amount of information collected at the sink is maximized and the total delay for the data aggregation is within the given bound. We formulate the problem by using dynamic programming and propose an optimal algorithm for the optimal assignment of transmission attempts. Based on the analysis of the optimal solution, we propose a distributed greedy algorithm. It is shown to have a similar performance as the optimal one."
    # keys = ["algorithms",
    #         "design",
    #         "performance",
    #         "sensor networks",
    #         "data aggregation",
    #         "real-time traffic",
    #         "csma/ca",
    #         "delay constrained transmission"]

    # sentence = abstract
    # keywords = keywords

    print("*" * 50)
    print("测试去除标点符号，统一小写")
    # 逐步执行预处理任务，以获得清理和归一化的文本语料库
    corpus = []
    keywords_clean = []

    # 转化为小写
    text_abstract = abstract.lower()
    # 去除空格，标签，引用
    text_abstract = text_abstract.strip()

    text_abstract = re.sub(r'[{}]+'.format('.!,;:?"()\''), '', text_abstract)
    text_abstract = re.sub("</?.*?>", "", text_abstract)
    # 去除关键词中存在的空格
    keywords = [' '.join(word.split()) for word in keywords if len(word) != 0 and word != " "]
    # text_abstract = re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}', '', text_abstract)
    # text_abstract = re.sub("(\d|\W)+", " ", text_abstract)
    # print(text)
    # 从string转为list
    # text = text.split()

    # 把每个词还原为原词 Lemmatisation

    abstract_lemmatisation = abstruct_pos_process(text_abstract)

    abstract_ready = [word for word in abstract_lemmatisation if not word in
                                                                     stop_words]
    print("现在把关键词矩阵也词性还原：")
    keywords_tagged_sent = pos_tag(keywords)  # 获取单词词性
    knl = WordNetLemmatizer()
    keywords_lemmatisation = []
    for key in keywords_tagged_sent:
        # print(key)
        keylist = key[0].split(" ")
        if len(keylist) == 1:
            # temp_str = []
            wordnet_pos_for_key = get_wordnet_pos(key[1]) or wordnet.NOUN
            print(key[0] + " : " + key[1] + " : " + wordnet_pos_for_key)
            keywords_lemmatisation.append(knl.lemmatize(key[0], pos=wordnet_pos_for_key))
        else:
            temp_str = []
            multi_keywords_tagged = pos_tag(keylist)  # 获取单词词性
            for oneword in multi_keywords_tagged:
                wordnet_pos_for_key = get_wordnet_pos(oneword[1]) or wordnet.NOUN
                temp_str.append(knl.lemmatize(oneword[0], pos=wordnet_pos_for_key))
            keywords_lemmatisation.append(temp_str)
    keywords_clean = keywords_lemmatisation

    corpus.append(abstract_ready)
    print(keywords_clean)

    print("-" * 50)

    keyword_long_list = []  # 长度大于1的关键词拆分list [['the', 'wall', 'street', 'journal'], ['apple', 'corporation']]
    for keyword in keywords_clean:
        if keyword.__len__() > 1:
            keyword_long_list.append(keyword)
    print(keyword_long_list)

    # abstracts_ready=[]
    #

    tags = []  # BIO序列
    print("===" * 50)
    print(abstract_ready)
    print(keywords_clean)
    skip_length = 0
    for i, word in enumerate(abstract_ready):  # 循环s1
        if skip_length > 0:
            skip_length = skip_length - 1
            print("skip->%s" % str(word))
            continue
        print(str(i) + " " + word)
        if word in keywords_clean:  # 如果关键词是一个词的，能直接找到并且打上标签B
            print("匹配到一个B")
            tags.append("B")
        else:
            length = multi_keys(word, keyword_long_list, i, abstract_ready)  # 求得匹配到的长度
            if length == -1:  # 如果匹配到的长度为-1说明长短词都不匹配 打O
                tags.append("O")
            else:
                tags.append("B")  # 否则先打B
                for x in range(length - 1):  # 其余长度为I
                    tags.append("I")
                skip_length = length - 1
                print("匹配到长序列，跳过%s个循环" % skip_length)

    print(abstract_ready)
    print(tags)
    return abstract_ready, tags


def abstract_keyword1(abstract, keywords):
    # 逐步执行预处理任务，以获得清理和归一化的文本语料库
    corpus = []
    keywords_clean = []

    # 去除关键词中存在的空格
    keywords = [' '.join(word.split()) for word in keywords if len(word) != 0 and word != " "]
    # 把每个词还原为原词 Lemmatisation
    print("现在把关键词矩阵也词性还原：")
    keywords_tagged_sent = pos_tag(keywords)  # 获取单词词性
    knl = WordNetLemmatizer()
    keywords_lemmatisation = []
    for key in keywords_tagged_sent:
        # print(key)
        keylist = key[0].split(" ")
        if len(keylist) == 1:
            # temp_str = []
            wordnet_pos_for_key = get_wordnet_pos(key[1]) or wordnet.NOUN
            print(key[0] + " : " + key[1] + " : " + wordnet_pos_for_key)
            keywords_lemmatisation.append(knl.lemmatize(key[0], pos=wordnet_pos_for_key))
        else:
            temp_str = []
            multi_keywords_tagged = pos_tag(keylist)  # 获取单词词性
            for oneword in multi_keywords_tagged:
                wordnet_pos_for_key = get_wordnet_pos(oneword[1]) or wordnet.NOUN
                temp_str.append(knl.lemmatize(oneword[0], pos=wordnet_pos_for_key))
            keywords_lemmatisation.append(temp_str)
    keywords_clean = keywords_lemmatisation
    # corpus.append(abstract_ready)
    print(keywords_clean)
    print("-" * 50)
    keyword_long_list = []  # 长度大于1的关键词拆分list [['the', 'wall', 'street', 'journal'], ['apple', 'corporation']]
    for keyword in keywords_clean:
        if keyword.__len__() > 1:
            keyword_long_list.append(keyword)
    print(keyword_long_list)

    # 转化为小写
    text_abstract = abstract.lower()
    # 去除空格，标签，引用
    text_abstract = text_abstract.strip()
    # 把一整个摘要拆分成多个句子，去除无关键词句子
    text_abstracts = text_abstract.split(".")

    short_abstracts = []
    short_abstracts_sentence = []
    short_abstracts_tag = []

    for sentence in text_abstracts:
        if len(sentence) == 0:
            continue
        sentence_clean = re.sub(r'[{}]+'.format('.!,;:?"()\''), '', sentence)
        sentence_clean = re.sub("</?.*?>", "", sentence_clean)
        sentence_lemmatisation = abstruct_pos_process(sentence_clean)
        abstract_ready = [word for word in sentence_lemmatisation if not word in stop_words]
        short_abstracts.append(abstract_ready)

    for short_abstract in short_abstracts:
        tags = []  # BIO序列
        print("===" * 50)
        print(short_abstract)
        print(keywords_clean)
        skip_length = 0
        has_keyword = 0
        for i, word in enumerate(short_abstract):  # 循环s1
            if skip_length > 0:
                skip_length = skip_length - 1
                print("skip->%s" % str(word))
                continue
            print(str(i) + " " + word)
            if word in keywords_clean:  # 如果关键词是一个词的，能直接找到并且打上标签B
                print("匹配到一个B")
                has_keyword = 1
                tags.append("B")
            else:
                length = multi_keys(word, keyword_long_list, i, short_abstract)  # 求得匹配到的长度
                if length == -1:  # 如果匹配到的长度为-1说明长短词都不匹配 打O
                    tags.append("O")
                else:
                    has_keyword = 1
                    tags.append("B")  # 否则先打B
                    for x in range(length - 1):  # 其余长度为I
                        tags.append("I")
                    skip_length = length - 1
                    print("匹配到长序列，跳过%s个循环" % skip_length)
        print(short_abstract)
        print(tags)
        if has_keyword == 1:
            short_abstracts_sentence.append(short_abstract)
            short_abstracts_tag.append(tags)

    cleaned_sentence = []
    cleaned_tag = []
    for sentence, tag in zip(short_abstracts_sentence, short_abstracts_tag):
        cleaned_sentence += sentence
        cleaned_tag += tag
    print(cleaned_sentence)
    print(cleaned_tag)
    return cleaned_sentence, cleaned_tag


def abstract_keyword2(abstract, keywords):
    # 逐步执行预处理任务，以获得清理和归一化的文本语料库
    corpus = []
    keywords_clean = []

    # 去除关键词尾存在的空格和空项
    keywords = [' '.join(word.split()) for word in keywords if len(word) != 0 and word != " "]
    # 把每个词还原为原词 Lemmatisation
    print("现在对关键词矩阵词性还原：")
    keywords_tagged_sent = pos_tag(keywords)  # 获取单词词性
    knl = WordNetLemmatizer()
    keywords_lemmatisation = []
    for key in keywords_tagged_sent:
        keylist = key[0].split(" ")
        if len(keylist) == 1:
            wordnet_pos_for_key = get_wordnet_pos(key[1]) or wordnet.NOUN
            print(key[0] + " : " + key[1] + " : " + wordnet_pos_for_key)
            keywords_lemmatisation.append(knl.lemmatize(key[0], pos=wordnet_pos_for_key))
        else:
            temp_str = []
            multi_keywords_tagged = pos_tag(keylist)  # 获取单词词性
            for oneword in multi_keywords_tagged:
                wordnet_pos_for_key = get_wordnet_pos(oneword[1]) or wordnet.NOUN
                temp_str.append(knl.lemmatize(oneword[0], pos=wordnet_pos_for_key))
            keywords_lemmatisation.append(temp_str)
    keywords_clean = keywords_lemmatisation
    # corpus.append(abstract_ready)
    # print(keywords_clean)
    # print("-" * 50)
    keyword_long_list = []  # 长度大于1的关键词拆分list [['the', 'wall', 'street', 'journal'], ['apple', 'corporation']]
    for keyword in keywords_clean:
        if keyword.__len__() > 1:
            keyword_long_list.append(keyword)
    print(keyword_long_list)

    # 转化为小写
    text_abstract = abstract.lower()
    # 去除空格，标签，引用
    text_abstract = text_abstract.strip()
    # 把一整个摘要拆分成多个句子，去除无关键词句子
    text_abstracts = text_abstract.split(".")

    short_abstracts = []
    short_abstracts_sentence = []
    short_abstracts_tag = []

    for sentence in text_abstracts:
        if len(sentence) == 0:
            continue
        sentence_clean = re.sub(r'[{}]+'.format('.!,;:?"()\''), '', sentence)
        sentence_clean = re.sub("</?.*?>", "", sentence_clean)
        sentence_clean = re.sub("\\(.* ?\\) |\\{. *?} | \\[. *?] | > | <", "", sentence_clean)
        # 关键词词性还原
        sentence_lemmatisation = abstruct_pos_process(sentence_clean)
        abstract_ready = [word for word in sentence_lemmatisation if not word in stop_words]
        short_abstracts.append(abstract_ready)

    for short_abstract in short_abstracts:
        tags = []  # 转化为BIO序列
        print("===" * 50)
        print(short_abstract)
        print(keywords_clean)
        skip_length = 0
        has_keyword = 0
        for i, word in enumerate(short_abstract):  # 循环s1
            if skip_length > 0:
                skip_length = skip_length - 1
                print("skip->%s" % str(word))
                continue
            print(str(i) + " " + word)
            if word in keywords_clean:  # 如果关键词是一个词的，能直接找到并且打上标签B
                print("匹配到一个B")
                has_keyword = 1
                tags.append("B")
            else:
                length = multi_keys(word, keyword_long_list, i, short_abstract)  # 求得匹配到的长度
                if length == -1:  # 如果匹配到的长度为-1说明长短词都不匹配 打O
                    tags.append("O")
                else:
                    has_keyword = 1
                    tags.append("B")  # 否则先打B
                    for x in range(length - 1):  # 其余长度为I
                        tags.append("I")
                    skip_length = length - 1
                    print("匹配到长序列，跳过%s个循环" % skip_length)
        print(short_abstract)
        print(tags)
        if has_keyword == 1:
            short_abstracts_sentence.append(short_abstract)
            short_abstracts_tag.append(tags)

    return short_abstracts_sentence, short_abstracts_tag
