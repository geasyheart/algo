# -*- coding: utf8 -*-
#
import random
from typing import List

import numpy as np


def random_sample_by_weight(l: List, p: List):
    """
    按照权重进行随机采样
    但是我看了下生成的结果，并不总是权重高的优先选择，毕竟是采样



    """
    if sum(p) == 0:
        return random.sample(l, 1)
    return np.random.choice(l, size=1, replace=False, p=p)


def random_walk(G, path_len: int, start_node=None):
    """
    随机游走
    """
    if start_node is None:
        # 随机采样一个起点
        path = random.sample(G.nodes, 1)
    else:
        path = [start_node]
    while len(path) < path_len:
        cur_node = path[-1]
        if len(G.adj[cur_node]) > 0:
            # ########################### 这里提供三种不同的策略进行采样 ############################
            # 1、不考虑边的权重
            # path.extend(random.sample(G.adj[cur_node].keys(), 1))
            # 2、考虑边的权重进行采样
            l, p = [], []
            for node, info in G.adj[cur_node].items():
                weight = info.get('weight', 0)
                l.append(node)
                p.append(weight)
            sum_p = sum(p)
            p = [i / sum_p for i in p]
            # print(p)
            path.extend(random_sample_by_weight(l=l, p=p))
            # 3、直接按照评分最高的进行随机游走, 注意，这样一定会生成大量重复数据，所以真实使用时不能按照这种方式
            # node_weights = [(node, info.get('weight', 0)) for node, info in G.adj[cur_node].items()]
            # print(node_weights)
            # node, highest_weight = sorted(node_weights, key=lambda x: x[1], reverse=True)[0]
            # path.append(node)
        else:
            break
    return path
