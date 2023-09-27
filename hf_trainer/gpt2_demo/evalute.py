# -*- coding: utf8 -*-
#
import json
import time

from transformers import GPT2Tokenizer, GPT2LMHeadModel

hf_model_path = 'checkpoint-27568'
tokenizer = GPT2Tokenizer.from_pretrained(hf_model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

model = GPT2LMHeadModel.from_pretrained(hf_model_path).to("cuda")

question = '这个想法可行吗' + "\n\n输出:"
inputs = tokenizer(question, return_tensors='pt').to("cuda")

res = model.generate(
    **inputs,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    max_length=128,
    # early_stopping=True,
    top_p=0.95,
    top_k=50,
    temperature=0.3,
    repetition_penalty=1.3,
)
print(tokenizer.decode(res[0]).split('<end>')[0].split('\n\n输出:')[1])


# ########################################3
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


s = time.time()
metric = QTDMetric()
counter = 0

with open('dev.jsonl.bak', 'r') as f:
    for index, line in enumerate(f, start=1):
        if index % 100 == 0:
            e = time.time()
            print(index, (e - s) / index)
        data = json.loads(line)
        question = data['input'] + "\n\n输出:"
        inputs = tokenizer(question, return_tensors='pt').to("cuda")
        res = model.generate(
            **inputs,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_length=128,
            early_stopping=True,
            top_p=0.95,
            top_k=50,
            temperature=0.3,
            repetition_penalty=1.3,
        )
        # print(tokenizer.decode(res[0]).split('<end>')[0])
        infer_text = tokenizer.decode(res[0]).split('<end>')[0].split('\n\n输出:')[1]

        metric.step(([infer_text], [data['output']]))
        print(infer_text, data['output'])
        counter += 1
e = time.time()
print(metric.report())
print((e - s) / counter)