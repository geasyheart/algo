import datasets
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
from transformers.utils import logging
from sklearn.metrics import classification_report, f1_score

model_checkpoint = 'chinese-roberta-wwm-ext'

train_file = 'train.jsonl'
dev_file = 'dev.jsonl'

raw_datasets = datasets.load_dataset('json', data_files={"train": train_file, "dev": dev_file})

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenizer_func(examples):
    bert_input = tokenizer.batch_encode_plus(
        examples['sent'],
        padding='longest', truncation=True, max_length=64,
        return_tensors='pt'
    )
    labels = torch.tensor(examples['label'], dtype=torch.long)
    return {
        **bert_input,
        "labels": labels
    }


train_dataset = raw_datasets['train'].map(tokenizer_func, batched=True)
dev_dataset = raw_datasets['dev'].map(tokenizer_func, batched=True)

logging.set_verbosity_info()

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

train_args = TrainingArguments(
    output_dir='./output',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy='steps',
    num_train_epochs=12,
    learning_rate=1e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    save_total_limit=1,
    load_best_model_at_end=True,
    save_steps=500,
    save_strategy='steps',
    metric_for_best_model='f1'
)
# train_args = TrainingArguments(
#     output_dir='./output',
#     overwrite_output_dir=True,
#     do_train=True,
#     do_eval=True,
#     evaluation_strategy='epoch',
#     num_train_epochs=12,
#     learning_rate=1e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     save_total_limit=1,
#     load_best_model_at_end=True,
#     save_strategy='epoch',
#     metric_for_best_model='f1'
# )
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_preds):
    preds = eval_preds.predictions.argmax(-1)
    labels = eval_preds.label_ids
    print(classification_report(y_true=labels, y_pred=preds, ))
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
