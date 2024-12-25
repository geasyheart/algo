import networkx as nx


def graph_cc():
    G = nx.Graph()

    # 添加节点
    G.add_nodes_from([1, 2, 3, 4, 5, 6])

    # 添加边
    G.add_edges_from([(1, 2), (2, 3), (4, 5), (5, 6)])

    # 查找连通分量
    components = nx.connected_components(G)

    # 打印每个连通分量
    for component in components:
        print(component)


def digraph_cc():
    import networkx as nx

    # 创建一个有向图
    G = nx.DiGraph()

    # 添加节点
    G.add_nodes_from([1, 2, 3, 4, 5, 6])

    # 添加有向边
    G.add_edges_from([(1, 2), (2, 3), (3, 1), (4, 5), (5, 6)])

    # 查找弱连通分量
    components = nx.weakly_connected_components(G)

    # 打印每个弱连通分量
    for component in components:
        print(component)


if __name__ == '__main__':
    digraph_cc()
