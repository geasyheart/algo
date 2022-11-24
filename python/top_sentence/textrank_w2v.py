# -*- coding: utf8 -*-
#

import json

import networkx as nx
import numpy as np

from utils import split_sentence

from gensim.models import Word2Vec
import LAC

# 拿测试的数据集训练一个word2vec，以这个获取词向量
model = Word2Vec.load("w2v.model")
docu_path = "/home/yuzhang/PycharmProjects/docu_words.txt"

lac = LAC.LAC(mode='rank')


def seg(sent: str):
    words, postags, ranks = lac.run(sent)
    return [(word, postag, rank) for word, postag, rank in zip(words, postags, ranks)]


def summarization(doc: str, topK=10):
    """
    这个准确来讲是返回doc中重要的句子
    """
    segs = []
    for sent in split_sentence(doc):
        if not sent.strip(): continue
        segs.append(seg(sent))

    graph = np.zeros((len(segs), len(segs)))

    for i in range(len(segs)):
        for j in range(i, len(segs)):
            cur_seg = segs[i]
            next_seg = segs[j]
            # lac中的词重要程度0,1,2,3
            rank_3_cur_words = [i[0] for i in cur_seg if i[2] >= 2]
            rank_3_next_words = [i[0] for i in next_seg if i[2] >= 2]
            if not rank_3_cur_words or not rank_3_next_words: continue
            # 这里采用word2vec来计算句子相似度
            score = model.n_similarity(ws1=rank_3_cur_words, ws2=rank_3_next_words)
            # 也可以尝试下bm25呢
            graph[i, j] = score
            graph[j, i] = score
    nx_graph = nx.from_numpy_matrix(graph)
    scores = nx.pagerank(nx_graph)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:topK]

    result = []
    # 再按照顺序恢复下
    # sorted_scores.sort(key=lambda x: x[0])
    for index, score in sorted_scores:
        result.append("".join([i[0] for i in segs[index]]))
    return result
