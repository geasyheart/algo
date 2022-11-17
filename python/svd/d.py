# -*- coding: utf8 -*-
#
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

df = pd.DataFrame({"军事": [0, 10, 7], '言情': [10, 3, 0], '动漫': [5, 3, 0], '小说': [3, 5, 0]}, dtype='float64')
#    军事    言情   动漫   小说
# 0   0.0  10.0  5.0  3.0
# 1  10.0   3.0  3.0  5.0
# 2   7.0   0.0  0.0  0.0
# 可以看到user0喜欢言情和动漫，用户1、2喜欢军事

U, s, Vt = svds(csc_matrix(df), k=2)
# U简单来说可以代表User的feature，Vt表示Item的feature。
matrix_new = U.dot(np.diag(s)).dot(Vt)
print(matrix_new)
# [[-0.1314746 ,  9.7949323 ,  5.08515349,  3.45149176],
# [10.28341738,  3.44206066,  2.81643619,  4.02672754],
# [ 6.55257518, -0.6978715 ,  0.28978818,  1.53648395]] # 推荐顺序-> 军事,小说,动漫

# ################################ 推荐
item_user_matrix = matrix_new.transpose()
svd_df = pd.DataFrame(item_user_matrix, columns=['user0', 'user1', 'user2'], index=df.columns)
svd_df['user2'].sort_values(ascending=False)


# Out[18]:
# 军事    6.552575
# 小说    1.536484
# 动漫    0.289788
# 言情   -0.697871
# Name: 2, dtype: float64

# ###################################### 计算用户喜好相似度
def cos(vec1, vec2):
    return vec1.dot(vec2) / np.linalg.norm(vec1) * np.linalg.norm(vec2)


for i in range(U.shape[0]):
    for j in range(i + 1, U.shape[0]):
        vec1 = U[i, :]
        vec2 = U[j, :]
        print(f'User{i}和User{j}的相似度:{cos(vec1, vec2)}')
# User0和User1的相似度:0.11017178104828797
# User0和User2的相似度:-0.11652231240752381
# User1和User2的相似度:0.28527719555421444
