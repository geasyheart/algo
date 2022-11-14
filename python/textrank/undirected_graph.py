# -*- coding: utf8 -*-
#
import networkx as nx

import numpy as np

# ##################################
# 一个是有向图，一个无向图
#
# ##################################
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
    # 这里哦，无向图构造
    matrix[v1_index][v2_index] = 1
    matrix[v2_index][v1_index] = 1
    # 这里哦，有向图构造
    matrix2[v1_index][v2_index] = 1

print(nx.pagerank(nx.from_numpy_matrix(matrix)))

print(nx.pagerank(nx.from_numpy_matrix(matrix2)))

