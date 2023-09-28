# -*- coding: utf8 -*-
#
# 其他的生成式模型都没有计算compute_metric，因为没法使用自定义的generation策略，所以此文同时结合chatGLM 的ptuning部分，来将这个过程梳理清除
# 参考https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py
import logging
import sys

import datasets
import numpy as np
import transformers
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, set_seed, T5Config, \
    HfArgumentParser, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments

pretrained_model_name_or_path = "mengzi-t5-base-mt"

train_data = 'train.jsonl'
dev_data = 'dev.jsonl'

logger = logging.getLogger(__name__)

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
def main():
    parser = HfArgumentParser(Seq2SeqTrainingArguments)
    training_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    raw_datasets = load_dataset(
        "json",
        data_files={"train": train_data, "validation": dev_data}
    )

    tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)

    prefix = '自己的prompt哦:'
    column_names = raw_datasets["train"].column_names

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

    if training_args.do_train:
        train_dataset = raw_datasets['train']
        train_dataset = train_dataset.select(range(16))
        with training_args.main_process_first(desc='train dataset map pre-processing'):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                desc='running tokenizer on train dataset'
            )
    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.select(range(16))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix='eval_t5', do_sample=True,
                                   max_length=64, top_p=0.95, top_k=50, temperature=0.3, repetition_penalty=1.3,
                                   early_stopping=True)
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    # 运行： --do_train --fp16 --output_dir=./output
    main()
