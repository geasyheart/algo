# -*- coding: utf8 -*-
#
import networkx as nx

# ############################## method1 ##############################

G = nx.DiGraph()
# 有向图之间边的关系
edges = [
    ("A", "B"), ("A", "C"), ("A", "A"), ("A", "A"), ("A", "D"),
    ("B", "A"), ("B", "C"),
    ("C", "D"), ("C", "B"),
    ("D", "B"), ("D", "A")
]
for edge in edges:
    G.add_edge(edge[0], edge[1])
pagerank_list = nx.pagerank(G, alpha=0.85)

print("pagerank1 -->", pagerank_list)

# ############################## method2 ##############################
# 使用矩阵导入的方式
G_matrix = nx.to_numpy_matrix(G)
print('注意这里，A->A是有两个的，但是没有权重哦：')
print(G_matrix)
G2 = nx.from_numpy_matrix(G_matrix, create_using=nx.DiGraph())

print("pagerank2 -->", nx.pagerank(G2, alpha=0.85))
