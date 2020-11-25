# -*- coding: utf8 -*-

# 参考地址: https://wax8280.github.io/2016/10/10/%E8%B0%88%E8%B0%88Kruskal%E4%B8%8EPrim%E8%BF%99%E4%B8%A4%E7%A7%8D%E6%9C%80%E5%B0%8F%E7%94%9F%E6%88%90%E6%A0%91%E7%AE%97%E6%B3%95%EF%BC%88Python%E5%AE%9E%E7%8E%B0%EF%BC%89/

# 算法是某个起始节点开始对目标图结构进行遍历，将其所有的出边加入到最小堆里面
# pop一条出边（每次pop的都是权值最小的边），然后判断这条边的目标点在生成树中
# 如果是就丢弃
# 不是就更新生成树

from heapq import heappop, heappush

G = {

    0: {},
    # source: {target: weight}
    1: {1: 1, 2: 1},
    2: {1: 1, 3: 6, 4: 11},
    3: {1: 2, 2: 6, 4: 9, 5: 13},
    4: {2: 11, 3: 9, 5: 7, 6: 3},
    5: {3: 13, 4: 7, 6: 4},
    6: {4: 3, 5: 4},
}


def prim(graph, start):
    """使用最小堆实现权重最小路径图"""
    P, Q = {}, [(0, None, start)]
    while Q:

        weight, source, target = heappop(Q)  # 获取weight最小的那个
        # 贪心匹配，如果存在，那么则忽略
        if target in P: continue  # 如果目标点在生成树中，跳过
        P[target] = source  # 记录目标点不在生成树中
        for point, weight in graph[target].items():
            heappush(Q, (weight, target, point))  # 将u点的出边入堆
    return P


T = prim(G, 1)
sum_count = 0
for k, v in T.items():
    if v is not None:
        sum_count += G[k][v]

print(sum_count)
print(T)
# 结果为19
# {1: None, 2: 1, 3: 1, 4: 3, 5: 6, 6: 4}
