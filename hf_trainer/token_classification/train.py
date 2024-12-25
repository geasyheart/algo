import json
import os.path

import datasets
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, \
    DataCollatorForTokenClassification
from transformers.utils import logging

os.environ["WANDB_DISABLED"] = "true"
# model_checkpoint = 'chinese-roberta-wwm-ext'
model_checkpoint = 'bert-base-chinese'

train_file = 'char_train.jsonl'
dev_file = 'char_dev.jsonl'

raw_datasets = datasets.load_dataset('json', data_files={"train": train_file, "dev": dev_file})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def get_label():
    label_id_map = {}
    with open(train_file, 'r') as f:
        for line in f:
            for label in json.loads(line)['labels']:
                label_id_map.setdefault(label, len(label_id_map))

    with open(os.path.join(os.path.dirname(__file__), 'label_id_map.json'), 'w') as f:
        f.write(json.dumps(label_id_map, ensure_ascii=False, indent=2))

    return label_id_map


LABEL_ID_MAP = get_label()


def tokenizer_func(examples):
    input_ids = [
        torch.LongTensor([tokenizer.cls_token_id, *tokenizer.convert_tokens_to_ids(_)[:510], tokenizer.sep_token_id])
        for _ in examples['tokens']
    ]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    token_type_ids = torch.zeros(size=input_ids.shape, dtype=torch.long)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    label_ids = [
        torch.LongTensor([
            -100,
            *[LABEL_ID_MAP[_] for _ in label][:510],
            -100,
        ])
        for label in examples['labels']
    ]
    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "labels": label_ids
    }


train_dataset = raw_datasets['train'].map(tokenizer_func, batched=True)
dev_dataset = raw_datasets['dev'].map(tokenizer_func, batched=True)

logging.set_verbosity_info()

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(LABEL_ID_MAP),
    id2label={v: k for k, v in LABEL_ID_MAP.items()},
    label2id=LABEL_ID_MAP,
)

# train_args = TrainingArguments(
#     output_dir='./output',
#     overwrite_output_dir=True,
#     do_train=True,
#     do_eval=True,
#     evaluation_strategy='steps',
#     num_train_epochs=12,
#     learning_rate=1e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     save_total_limit=1,
#     load_best_model_at_end=True,
#     save_steps=30,
#     save_strategy='steps',
#     metric_for_best_model='f1'
# )
train_args = TrainingArguments(
    output_dir='./output',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    num_train_epochs=12,
    learning_rate=1e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    save_total_limit=1,
    load_best_model_at_end=True,
    save_strategy='epoch',
    metric_for_best_model='f1'
)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


def compute_metrics(eval_preds):
    preds = eval_preds.predictions.argmax(-1)
    labels = eval_preds.label_ids

    mask = np.not_equal(labels, -100)
    labels = labels[mask]
    preds = preds[mask]

    print(classification_report(y_true=labels, y_pred=preds))
    return {
        "f1": f1_score(y_true=labels, y_pred=preds, average='macro')
    }


trainer = Trainer(
    model=model,
    args=train_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)
trainer.train()
