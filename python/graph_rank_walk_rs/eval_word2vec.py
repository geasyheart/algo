# -*- coding: utf8 -*-
#
import json
import pathlib

import pandas as pd
import torch
import tqdm
from torch import nn
from torch.functional import F


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size):
        super(SkipGramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.output = nn.Linear(64, vocab_size)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        output = self.output(embed)
        return F.log_softmax(output, dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("unique_id.json", "r") as f:
    unique_id_map = json.loads(f.read())

model = SkipGramModel(len(unique_id_map))
model.to(device)
model.load_state_dict(torch.load("skipgram.pt"))

# ############################ eval

movies_path = pathlib.Path(__file__).parent.joinpath('data').joinpath('movies.csv')
ratings_path = pathlib.Path(__file__).parent.joinpath('data').joinpath('ratings.csv')

movies_df = pd.read_csv(movies_path)
# 过滤掉title为null的
movies_df = movies_df[~pd.isnull(movies_df.title)]
rating_df = pd.read_csv(ratings_path)
rating_df = rating_df[rating_df['movieId'].isin(movies_df['movieId'])]

id_unique_map = {v: k for k, v in unique_id_map.items()}


def view(userid):
    # 找到评过分的电影
    movie_ids = set(rating_df[rating_df['userId'] == userid]['movieId'].tolist())
    # 模型推荐下
    unique_id = unique_id_map[f'user:{userid}']
    output = model(torch.tensor([unique_id]).to(device))
    topK = output[0].argsort(descending=True).tolist()

    recommend_result = {"movie": set(), "user": set()}
    for topid in topK:
        item: str = id_unique_map[topid]
        if item.startswith('movie:'):
            _, id = item.split('movie:')
            recommend_result["movie"].add(int(id))

            if len(recommend_result["movie"]) == len(movie_ids):
                break

        elif item.startswith('user:'):
            _, id = item.split('user:')
            recommend_result["user"].add(int(id))

    pred_same_size = len(movie_ids & recommend_result['movie'])
    return len(movie_ids), pred_same_size


def eval_result():
    rate_size, recommend_same_size = 0, 0
    for userId in tqdm.tqdm(rating_df['userId'].unique().tolist(), total=len(rating_df['userId'].unique().tolist())):
        had_rate_movies, recommend_movies = view(userid=userId)
        rate_size += had_rate_movies
        recommend_same_size += recommend_movies
    # 随机 random walk
    # 0.3500782151213342
    # 直接按照权重最高的进行random walk
    # 0.053764018740918885
    # 从而可以看出，在不优化代码的前提下，更改随机游走方式是可以影响最终评判结果的
    print(recommend_same_size / rate_size)


if __name__ == '__main__':
    eval_result()
