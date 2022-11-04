# -*- coding: utf8 -*-
#
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

documents = ['我 爱 你 我', '我 爱 中国', '我 爱 祖国']
vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split())
transformer = TfidfTransformer()

tfidf_vec = transformer.fit_transform(vectorizer.fit_transform(documents))

article_keywords = vectorizer.get_feature_names()
article_keyword_weights = tfidf_vec.toarray()

# 获取每个document的tfidf
article_keyword_weight_df = pd.DataFrame(article_keyword_weights, columns=article_keywords)
print(article_keyword_weight_df)

# 获取idf表
idf = pd.DataFrame(transformer.idf_.reshape(1, -1), columns=article_keywords)
print(idf)
