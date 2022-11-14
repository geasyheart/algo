# -*- coding: utf8 -*-
#
import networkx as nx


def weight1():
    G = nx.DiGraph()
    G.add_edge('A', 'B', weight=3)
    G.add_edge('A', 'C')
    # print(nx.to_numpy_matrix(G))
    return nx.pagerank(G)


def without_weight():
    G = nx.DiGraph()
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    # print(nx.to_numpy_matrix(G))
    return nx.pagerank(G)


if __name__ == '__main__':
    # 这里结果是不一样滴，print下matrix就知道了
    print(weight1())
    print(without_weight())
