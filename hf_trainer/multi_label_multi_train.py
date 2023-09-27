# -*- coding: utf8 -*-
#

import json
from typing import Dict

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import numpy as np

from umetrics.multi_label_class_metric import MultiLabelClassMacroF1Metric

transformers.logging.set_verbosity_info()
print(torch.cuda.is_available())


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


train_data, dev_data = [], []
with open('train.jsonl', 'r') as f:
    for line in f:
        train_data.append(json.loads(line))

with open('dev.jsonl', 'r') as f:
    for line in f:
        dev_data.append(json.loads(line))

LABEL_ID_MAP = {"时间": 0, "空间": 1, '数量': 2}
pretrained_model_or_path = "chinese-roberta-wwm-ext"

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_or_path, num_labels=len(LABEL_ID_MAP))
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_or_path)


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def collate_fn(batch_data):
    bt_text = []
    bt_labels = torch.zeros(len(batch_data), len(LABEL_ID_MAP))
    for index, data in enumerate(batch_data):
        bt_text.append(data['query'])
        # bt_labels.append([LABEL_ID_MAP[_] for _ in data['labels']])
        for label in data['labels']:
            label_id = LABEL_ID_MAP[label]
            bt_labels[index][label_id] = 1
    data = tokenizer.batch_encode_plus(bt_text, return_tensors='pt', padding=True, truncation=True, max_length=510)
    data.update({"labels": bt_labels})
    return data


def compute_metrics(data) -> Dict:
    bt_preds = (sigmoid(data.predictions) > 0.5).astype(float).tolist()
    bt_labels = data.label_ids.tolist()

    mm = MultiLabelClassMacroF1Metric()
    mm.label_map = LABEL_ID_MAP
    mm.step(preds=bt_preds, labels=bt_labels)
    mm.report()
    return {"f1": mm.score()}


# args = TrainingArguments(
#     output_dir='./output',
#     remove_unused_columns=False,
#     seed=1000,
#     do_train=True, do_eval=True,
#     evaluation_strategy='epoch',
#
#     learning_rate=1e-5,
#     num_train_epochs=4,
#     weight_decay=0.01,
#     per_device_train_batch_size=64,
#     fp16=True,
#     save_total_limit=2,
#     load_best_model_at_end=True,
#     save_strategy='epoch',
#     metric_for_best_model='f1'
# )

args = TrainingArguments(
    output_dir='./output',
    remove_unused_columns=False,
    seed=1000,
    do_train=True, do_eval=True,
    evaluation_strategy='steps',
    learning_rate=1e-5,
    num_train_epochs=4,
    weight_decay=0.01,
    per_device_train_batch_size=64,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    save_strategy='steps',
    save_steps=100,
    metric_for_best_model='f1'
)

trainer = Trainer(
    model=model,
    args=args,
    data_collator=collate_fn,
    train_dataset=MyDataset(train_data),
    eval_dataset=MyDataset(train_data),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
