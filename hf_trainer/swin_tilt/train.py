import os
import pprint
import random
import re
from collections import defaultdict

import torch
import torch.nn as nn
import tqdm
from PIL import Image
from datasets import Dataset as HFDataset
from sklearn.metrics import classification_report, f1_score
from torch.nn import BCELoss
from torch.utils.data import Dataset
from torchvision import transforms
# 加载 SwinTransformer 模型和处理器
from transformers import (AutoFeatureExtractor, SwinModel, Trainer,
                          TrainingArguments)
from transformers.models.swin.modeling_swin import SwinImageClassifierOutput

os.environ["WANDB_DISABLED"] = "true"
MODEL_NAME_OR_PATH = './swin-large-patch4-window12-384-in22k/'

transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(384),
])


def load_dataset_from_path():
    result = []
    image_dirs = ['./images', ]
    for image_dir in image_dirs:
        for image_name in tqdm.tqdm(os.listdir(image_dir), total=len(os.listdir(image_dir)), postfix=image_dir):
            label = re.search(r"l\-(是|否)\.", image_name).group(1)
            image_path = os.path.join(image_dir, image_name)
            try:
                transform(Image.open(image_path).convert("RGB"))
            except Exception as e:
                print(f"invalid image: {image_path}, err:{e}")
                continue

            result.append({"image_path": image_path, "label": label})

    # #################### split
    random.shuffle(result)

    counter = defaultdict(int)
    for item in result:
        counter[item['label']] += 1
    pprint.pprint(counter)

    trains, tests = [], []
    train_counter = defaultdict(int)
    for item in result:
        if train_counter[item['label']] / counter[item['label']] > 0.85:
            tests.append(item)
        else:
            train_counter[item['label']] += 1
            trains.append(item)
    return HFDataset.from_list(trains), HFDataset.from_list(tests)
    # all_dataset = HFDataset.from_list(result)
    #
    # split = all_dataset.train_test_split(0.15, seed=1)
    # return split['train'], split['test']


train_dataset, eval_dataset = load_dataset_from_path()
print(
    f'total train size:{train_dataset.shape[0]},eval size:{eval_dataset.shape[0]}')


class SwinForClassify(nn.Module):
    def __init__(self, model_name="microsoft/swin-large-patch4-window12-384-in22k", num_labels=1):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(model_name)  # 只加载特征提取部分
        hidden_size = self.backbone.config.hidden_size  # Swin的最后一层特征维度
        self.classify = nn.Linear(hidden_size, num_labels)  # 线性回归层

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values)  # 提取特征
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        logits = torch.sigmoid(self.classify(pooled_output))  # 线性映射到目标维度
        loss = None
        if labels is not None:
            cri = BCELoss()
            loss = cri(logits, labels)

        return SwinImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,

        )


# 示例
model = SwinForClassify(model_name=MODEL_NAME_OR_PATH, num_labels=1)

processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME_OR_PATH)


class ClassifyDataset(Dataset):
    def __init__(self, dataset, processor, transform=None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image_path']
        image = Image.open(image).convert('RGB')
        image = transform(image)

        label = self.dataset[idx]['label']
        if label == '是':
            label = 1
        elif label == '否':
            label = 0
        else:
            raise ValueError(f'{label} not valid.')
        return {"image": image, "label": label}


# 创建数据集对象
train_dataset = ClassifyDataset(train_dataset, processor)
eval_dataset = ClassifyDataset(eval_dataset, processor)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",  # 输出结果目录
    overwrite_output_dir=True,
    num_train_epochs=10,  # 训练轮数
    per_device_train_batch_size=4,  # 每设备训练的batch size
    per_device_eval_batch_size=8,  # 每设备评估的batch size
    evaluation_strategy="epoch",  # 每个epoch评估一次
    save_strategy="epoch",  # 每个epoch保存一次
    logging_dir='./logs',  # 日志目录
    logging_steps=10,
    learning_rate=2e-5,  # 学习率
    weight_decay=0.01,  # 权重衰减
    load_best_model_at_end=True,  # 训练结束时加载最好的模型
    remove_unused_columns=False,
    metric_for_best_model='f1',
)


def data_collator(batch):
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    inputs = processor(images, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels, dtype=torch.float).reshape(-1, 1)
    return inputs


def compute_metrics(eval_preds):
    preds, labels = [], []
    acc, total = 0, 0

    for (pred, label) in zip(eval_preds.predictions, eval_preds.label_ids):
        if pred > 0.5:
            pred = 1
        else:
            pred = 0
        preds.append(pred)
        labels.append(label)

        total += 1
        if pred == label:
            acc += 1
    print(f'ACC: {acc / (total + 1e-5)}')
    print(classification_report(y_true=labels, y_pred=preds, ))
    return {"f1": f1_score(y_true=labels, y_pred=preds, average='macro')}


# 定义 Trainer
trainer = Trainer(
    model=model,  # 预训练模型
    args=training_args,  # 训练参数
    train_dataset=train_dataset,  # 训练数据集
    eval_dataset=eval_dataset,  # 验证数据集
    tokenizer=processor,  # 使用Swin图像处理器
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# 训练模型
# trainer.train(resume_from_checkpoint=True)
trainer.train()

# 保存模型
trainer.save_model("./final_model")
