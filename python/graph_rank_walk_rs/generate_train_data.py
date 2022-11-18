# -*- coding: utf8 -*-
#
import json
import pathlib

import networkx as nx
import pandas as pd
import tqdm

from python.graph_rank_walk_rs.rank_walk import random_walk

movies_path = pathlib.Path(__file__).parent.joinpath('data').joinpath('movies.csv')
ratings_path = pathlib.Path(__file__).parent.joinpath('data').joinpath('ratings.csv')

movies_df = pd.read_csv(movies_path)
# 过滤掉title为null的
movies_df = movies_df[~pd.isnull(movies_df.title)]
print(movies_df.shape)

rating_df = pd.read_csv(ratings_path)
print(rating_df.shape)
# 只考虑有title的评分
rating_df = rating_df[rating_df['movieId'].isin(movies_df['movieId'])]
print(rating_df.shape)

# 给每个用户和movie分配一个唯一ID
unique_id = {}

# 构造graph
G = nx.Graph()

for index, row in tqdm.tqdm(rating_df.iterrows(), total=rating_df.shape[0], desc='build data'):
    userId = row['userId']
    movieId = row['movieId']
    unique_id.setdefault(f'user:{userId}', len(unique_id))
    unique_id.setdefault(f'movie:{movieId}', len(unique_id))
    G.add_edge(f'user:{userId}', f'movie:{movieId}', weight=row['rating'])  # 这里没有考虑rating weight

# 随机游走生成训练数据
with open("unique_id.json", "w") as f:
    f.write(json.dumps(unique_id))

with open("data.jsonl", 'w') as f:
    total_size = 500000
    for i in tqdm.tqdm(range(total_size), total=total_size, desc='generate data'):
        path = random_walk(G, path_len=5)
        f.write(json.dumps(path) + "\n")
