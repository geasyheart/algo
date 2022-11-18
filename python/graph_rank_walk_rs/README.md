## 说明

基于word2vec的方式来训练graph embedding

有users以及movies，通过rating(打分)进行关联到一起，只要有打分，就说明用户看过这个电影


## 运行
0. 按照data/README.md下载数据
1. 运行generate_train_data.py，生成训练数据
2. 运行train_word2vec.py
3. 运行eval_word2vec.py

## 注意

没有考虑权重（此处是rating），结果是0.3500782151213342
