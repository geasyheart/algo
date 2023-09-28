# -*- coding: utf8 -*-
#
import logging

from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, GenerationConfig
from transformers.utils import logging
import numpy as np

pretrained_model_name_or_path = "mengzi-t5-base-mt"

train_data = '/data/.tmpyz/train.jsonl'
dev_data = '/data/.tmpyz/dev.jsonl'

# ########################################
import datetime
from typing import Dict

from LAC import LAC
from umetrics.bleu import BLEUMetric
from utrainer.metric import Metric

lac = LAC()


class QTDMetric(Metric):
    def __init__(self):
        self.bleu = BLEUMetric()

        self.correct = 0
        self.total = 0

    def step(self, inputs):
        y_preds, y_trues = inputs
        # add acc
        self.total += len(y_preds)
        for y_pred, y_true in zip(y_preds, y_trues):
            if y_pred == y_true:
                self.correct += 1

        # 增加一步
        not_corrects = []
        for y_pred, y_true in zip(y_preds, y_trues):
            if y_pred != y_true:
                not_corrects.append((y_pred, y_true))
        if 0 < len(not_corrects) / len(y_preds) <= 0.5:
            now = str(datetime.datetime.now())
            with open("/tmp/err.log", "a+") as f:
                for (y_pred, y_true) in not_corrects:
                    f.write(f'{now}\t{y_pred}\t{y_true}\n')

        # add bleu
        y_preds = lac.run(y_preds)
        y_trues = lac.run(y_trues)
        y_preds = [[single[0]] for single in y_preds]
        y_trues = [single[0] for single in y_trues]
        self.bleu.step(y_trues=y_trues, y_preds=y_preds)

    def score(self) -> float:
        return self.bleu.score()

    def report(self) -> Dict:
        print(self.bleu.report())
        acc = self.correct / (self.total + 1e-5)
        print(f'总样本:{self.total},正确:{self.correct},Acc:{acc}')
        return {}


# ###########################################
logging.set_verbosity_info()
model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path)
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    generation_config=GenerationConfig(
        do_sample=True,
        max_length=64,
        top_p=0.95,
        top_k=50, temperature=0.3, repetition_penalty=1.3,
        early_stopping=True,
        decoder_start_token_id=model.config.decoder_start_token_id
    ),
    output_dir='./output',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy='steps',
    num_train_epochs=12,
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    fp16=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    save_strategy='steps',
    save_steps=500,
    metric_for_best_model='bleuAvg'

)
raw_datasets = load_dataset(
    "json",
    data_files={"train": train_data, "validation": dev_data}
)
train_dataset = raw_datasets['train']
dev_dataset = raw_datasets['validation']

prefix = '自己的prompt哦:'


def preprocess_function(examples):
    inputs, targets = [], []
    for i in range(len(examples['input'])):
        if examples['input'][i] and examples['output'][i]:
            inputs.append(examples['input'][i])
            targets.append(examples['output'][i])
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=64, padding="longest", truncation=True)
    labels = tokenizer(targets, max_length=64, padding='longest', truncation=True)
    labels['input_ids'] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


column_names = raw_datasets["train"].column_names

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=column_names,
    desc='running tokenizer on train dataset'
)
dev_dataset = dev_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=column_names,
    desc='running tokenizer on eval dataset'
)

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    metric = QTDMetric()
    metric.step([decoded_preds, decoded_labels])
    metric.report()
    return {"bleuAvg": metric.score()}


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,

)
trainer.train()
