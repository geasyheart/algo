# -*- coding: utf8 -*-
#


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
# print(torch.__version__)

from collections import Counter
import numpy as np
import random


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE

import scipy
import thulac


#
# 1. download addr: http://down.xshuyaya.cc/zip/%E5%89%91%E6%9D%A5.zip
# 2. 分词
thu1 = thulac.thulac(seg_only=True)
thu1.cut_f('剑来.txt', 'p_output.txt')

# 3. 预处理
# 标点符号集
stopwords = '''~!@#$%^&*()_+`1234567890-={}[]:：";'<>,.?/|\、·！（）￥“”‘’《》，。？/—-【】….'''
stopwords_set = set([i for i in stopwords])
stopwords_set.add("br") # 异常词也加入此集，方便去除

data = []

with open('p_output.txt', 'r') as f:
    for line in f:
        for s in stopwords_set:
            line = line.strip().replace(s, '')
        line = line.replace("   ", " ").replace("  ", " ")
        if line != "" and line != " ":
            data.append(line)


# 保存数据
with open("all.txt", "w") as f:
    f.write(" ".join(data))


# 生成数据集

# 分割数据为训练集、测试集和验证集，并保存
all_text = ""
with open("all.txt", "r", ) as f:
    all_text = f.readline()

all_len = len(all_text)
train_text = all_text[:int(all_len * 0.9)]
dev_text = all_text[int(all_len * 0.9):int(all_len * 0.95)]
test_text = all_text[int(all_len * 0.95):]

with open("dev.txt", "w", ) as f:
    f.write(dev_text)
with open("test.txt", "w", ) as f:
    f.write(test_text)
with open("train.txt", "w", ) as f:
    f.write(train_text)