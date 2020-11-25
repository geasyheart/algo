# -*- coding: utf8 -*-


# 数据一样
import heapq

G = {
    0:{},
    1: {1: 1, 2: 1},
    2: {1: 1, 3: 6, 4: 11},
    3: {1: 2, 2: 6, 4: 9, 5: 13},
    4: {2: 11, 3: 9, 5: 7, 6: 3},
    5: {3: 13, 4: 7, 6: 4},
    6: {4: 3, 5: 4},
}

# 这里将权重变成负数，那么此时就变成了最大堆的方式
for key, value in G.items():
    for _key, _value in value.items():
        value[_key] = 0 - _value

print(G)


def prim_by_sort(graph, start):
    P, Q = {}, [(0, None, start)]
    while Q:

        weight, source, target = heapq.heappop(Q)  # 获取weight最小的那个
        # 贪心匹配，如果存在，那么则忽略
        if target in P: continue  # 如果目标点在生成树中，跳过
        P[target] = source  # 记录目标点不在生成树中
        for point, weight in graph[target].items():
            heapq.heappush(Q, (weight, target, point))  # 将u点的出边入堆
    return P


T = prim_by_sort(G, 1)
# 这里sum_count就变成了负值，所以需要减
sum_count = 0
for k, v in T.items():
    if v is not None:
        sum_count -= G[k][v]

print(sum_count)
print(T)
