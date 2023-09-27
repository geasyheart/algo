# -*- coding: utf8 -*-
#
import json

from torch.utils.data import Dataset
from transformers import BloomTokenizerFast, BloomForCausalLM, TrainingArguments, Trainer
from transformers.utils import logging

model_name_or_path = 'bloom-389m-zh'
tokenizer = BloomTokenizerFast.from_pretrained(model_name_or_path)
model = BloomForCausalLM.from_pretrained(model_name_or_path)


class Q2SDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def data_collator(bt):
    bt_inputs = []
    for _ in bt:
        # NOTE: 如果不添加eos符，效果不是很好哦
        bt_inputs.append('自己添加一个prompt哦:' + _['input'] + "\n\n输出:" + _['output'])
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
    output_dir='bloom_output',
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

train_path = 'train.jsonl'
dev_path = 'dev.jsonl'


def read_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


train_data, dev_data = list(read_data(train_path)), list(read_data(dev_path))

trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    train_dataset=Q2SDataset(train_data),
    eval_dataset=Q2SDataset(dev_data),
    tokenizer=tokenizer,

)
trainer.train()
