# -*- coding: utf8 -*-
#
import networkx as nx

from rank_walk import random_walk


def undirected_graph():
    G = nx.Graph()

    G.add_node('A')
    G.add_node('B')
    G.add_node('C')
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('B', 'C')
    # A -> B
    # A -> C
    # B -> C

    print(G.nodes)
    for node in ['A', 'B', 'C']:
        print(G.adj[node])

    print(random_walk(G, path_len=4))


def directed_graph():
    G = nx.DiGraph()
    G.add_node('A')
    G.add_node('B')
    G.add_node('C')
    G.add_edge('A', 'B', weight=3)
    G.add_edge('A', 'C', weight=1)
    G.add_edge('B', 'C')
    # A -> B
    # A -> C
    # B -> C

    print(G.nodes)
    for node in ['A', 'B', 'C']:
        print(G.adj[node])

    print(random_walk(G, path_len=4))


if __name__ == '__main__':
    # undirected_graph()
    directed_graph()
