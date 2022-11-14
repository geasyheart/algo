# -*- coding: utf8 -*-
#
import networkx as nx


def textrank1():
    """
    有向图之间边的关系
    """
    G = nx.DiGraph()
    edges = [
        ("A", "B"), ("A", "C"), ("A", "D"),
        ("B", "A"), ("B", "C"),
        ("C", "D"), ("C", "B"),
        ("D", "B"), ("D", "A")
    ]
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    pagerank_list = nx.pagerank(G, alpha=0.85)
    return pagerank_list, nx.to_numpy_matrix(G)


def textrank2(matrix):
    """
        使用矩阵导入的方式
        我这里是有向图哦
    """
    G2 = nx.from_numpy_matrix(matrix, create_using=nx.DiGraph())
    return nx.pagerank(G2, alpha=0.85)


if __name__ == '__main__':
    r1, matrix = textrank1()
    r2 = textrank2(matrix)
    print(r1)
    print(r2)
