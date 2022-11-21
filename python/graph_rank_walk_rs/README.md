## 说明

基于word2vec的方式来训练graph embedding

有users以及movies，通过rating(打分)进行关联到一起，只要有打分，就说明用户看过这个电影


## 运行
0. 按照data/README.md下载数据
1. 运行generate_train_data.py，生成训练数据
2. 运行train_word2vec.py
3. 运行eval_word2vec.py

## 注意

1. 没有考虑权重（此处是rating），结果是0.3500782151213342
2. 按照权重进行random walk，这个方式没试（时间）
3. 直接按照权重最高的进行random walk，结果是0.053764018740918885

结论，如果考虑泛化性的话，可以考虑第一种。
第二种是考虑了权重，那么评分高的电影也会被优先推荐出来。
第三种只是用于验证这套方式的可行性，改动了random walk数据生成方式，**不建议使用**。

