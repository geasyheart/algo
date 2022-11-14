# -*- coding: utf8 -*-
#

# 这里我发现一个有意思的问题，就是在构造matrix和matrix2的时候，看下面
import networkx as nx

import numpy as np

# ##################################
# 但是我见到一种比较emm的方式，
edges = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G')]
matrix = np.zeros(shape=(7, 7))
matrix2 = np.zeros(shape=(7, 7))
node_id_map = {}
for v1, v2 in edges:
    node_id_map.setdefault(v1, len(node_id_map))
    node_id_map.setdefault(v2, len(node_id_map))
print(node_id_map)

for v1, v2 in edges:
    v1_index = node_id_map[v1]
    v2_index = node_id_map[v2]
    # 这里哦
    matrix[v1_index][v2_index] = 1
    matrix[v2_index][v1_index] = 1
    # 这里哦
    matrix2[v1_index][v2_index] = 1

print(nx.pagerank(nx.from_numpy_matrix(matrix)))

print(nx.pagerank(nx.from_numpy_matrix(matrix2)))
# 会发现这俩结果是一样的，这是一个非常神奇的现象，不知道底层怎么实现的。。这个是在TextRank4zh中发现的
