# -*- coding: utf8 -*-
#
import os

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import evaluate
import numpy as np

os.environ['http_proxy'] = 'http://192.168.0.76:1080'
os.environ['https_proxy'] = 'http://192.168.0.76:1080'

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-chinese"  # "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer", evaluation_strategy='epoch')


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,

    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)


trainer.train()


# predictions = trainer.predict(tokenized_datasets["validation"])
# print(predictions.predictions.shape, predictions.label_ids.shape)

