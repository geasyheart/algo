# -*- coding: utf8 -*-
#
import json

from torch.utils.data import Dataset
from transformers import Trainer, GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, DataCollatorForLanguageModeling, \
    logging

logging.set_verbosity_info()

train_path = 'train.jsonl'
dev_path = 'dev.jsonl'


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


train_data, dev_data = list(read_data(train_path)), list(read_data(dev_path))
# train_data = train_data[:1000]
# dev_data = dev_data[:1000]
hf_model_path = 'Wenzhong-GPT2-110M'
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
model = GPT2LMHeadModel.from_pretrained(hf_model_path)


class Q2SDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})


def data_collator(bt):
    bt_inputs = []
    for _ in bt:
        bt_inputs.append(_['input'] + "\n\n输出:" + _['output'] + '<end>')
    inputs_dict = tokenizer(bt_inputs, padding=True, truncation=True, max_length=128, return_tensors='pt')
    target = inputs_dict['input_ids']
    labels = target.clone().detach()
    labels[target == tokenizer.pad_token_id] = -100
    return {
        "input_ids": inputs_dict['input_ids'],
        'attention_mask': inputs_dict['attention_mask'],
        'labels': labels
    }


args = TrainingArguments(
    output_dir='output',
    overwrite_output_dir=True,
    do_train=True,
    # do_eval=True,
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=2,
    num_train_epochs=8,
    learning_rate=1e-4,
    save_strategy='epoch',
    save_total_limit=3,
    fp16=True,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    warmup_ratio=0.01,
    weight_decay=0.1

)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=Q2SDataset(train_data),
    eval_dataset=Q2SDataset(dev_data),
    tokenizer=tokenizer,

)
trainer.train()
# NOTE: 自己在input那里添加prompt哦
