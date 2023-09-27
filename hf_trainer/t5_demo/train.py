# -*- coding: utf8 -*-
#
import json

from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from transformers.utils import logging

train_path = 'train.jsonl'
dev_path = 'dev.jsonl'


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


train_data, dev_data = list(read_data(train_path)), list(read_data(dev_path))

model_name_or_path = 'mengzi-t5-base-mt'
tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)


class Q2SDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def data_collator(bt):
    bt_inputs, bt_labels = [], []
    for _ in bt:
        bt_inputs.append("自己添加一个prompt哦:" + _['input'])
        bt_labels.append(_['output'])
    inputs_dict = tokenizer(bt_inputs, padding=True, truncation=True, max_length=64, return_tensors='pt')
    label_dict = tokenizer(bt_labels, padding=True, truncation=True, max_length=64, return_tensors='pt')
    label_ids = label_dict['input_ids']
    label_ids[label_ids == tokenizer.pad_token_id] = -100
    inputs_dict.update({'labels': label_ids})
    return inputs_dict


args = TrainingArguments(
    output_dir='t5_output',
    overwrite_output_dir=True,
    do_train=True,
    # do_eval=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=2,
    num_train_epochs=8,
    learning_rate=1e-4,
    save_strategy='epoch',
    save_total_limit=3,
    fp16=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,

)
logging.set_verbosity_info()


# train_data = train_data[:100]
# dev_data = dev_data[:100]


def compute_metrics(bt):
    print()


trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=Q2SDataset(train_data),
    eval_dataset=Q2SDataset(dev_data),
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)
trainer.train()
