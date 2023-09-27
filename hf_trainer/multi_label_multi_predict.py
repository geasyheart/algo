# -*- coding: utf8 -*-
#
import json

import torch
from sklearn.metrics import f1_score, classification_report
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# pipe = pipeline(
#     task="text-classification",
#     model="output/checkpoint-532",
#     device=torch.device("cuda:0"),
# )
tokenizer = AutoTokenizer.from_pretrained('checkpoint-532')
model = AutoModelForSequenceClassification.from_pretrained('checkpoint-532')

res = tokenizer('今天哪里下雨,啥时候下雨', return_tensors='pt')
aa = model(**res)
print(res)