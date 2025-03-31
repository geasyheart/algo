import os
import pprint
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import tqdm
from PIL import Image
from datasets import Dataset as HFDataset
from sklearn.metrics import classification_report, f1_score
from torch.nn import MSELoss
from torch.utils.data import Dataset
from torchvision import transforms
# 加载 SwinTransformer 模型和处理器
from transformers import (AutoFeatureExtractor, SwinModel, Trainer,
                          TrainingArguments)
from transformers.models.swin.modeling_swin import SwinImageClassifierOutput

os.environ["WANDB_DISABLED"] = "true"
MODEL_NAME_OR_PATH = './swin-large-patch4-window12-384-in22k/'
# MODEL_NAME_OR_PATH = './swin-base-patch4-window12-384-in22k'


transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
])


def angle_to_target(angle):
    # angle_rad = angle * np.pi / 180
    angle_rad = np.radians(angle)
    sin_angle = np.sin(angle_rad)
    cos_angle = np.cos(angle_rad)
    return sin_angle, cos_angle


def pred_to_angle(sin_pred, cos_pred):
    angle_rad = np.arctan2(sin_pred, cos_pred)
    angle_deg = angle_rad * 180.0 / np.pi
    # 确保角度在 0°~360°
    angle_deg = (angle_deg + 360.0) % 360.0
    return angle_deg


def load_dataset_from_path():
    result = []
    image_dirs = ['./images', ]
    for image_dir in image_dirs:
        for image_name in tqdm.tqdm(os.listdir(image_dir), total=len(os.listdir(image_dir)), postfix=image_dir):
            angle = int(re.search(r"angle\-(\d+)\.", image_name).group(1))
            image_path = os.path.join(image_dir, image_name)

            result.append({"image_path": image_path, "angle": angle})

    # #################### split
    random.shuffle(result)

    counter = defaultdict(int)
    for item in result:
        counter[item['angle']] += 1
    pprint.pprint(counter)

    trains, tests = [], []
    train_counter = defaultdict(int)
    for item in result:
        if train_counter[item['angle']] / counter[item['angle']] > 0.85:
            tests.append(item)
        else:
            train_counter[item['angle']] += 1
            trains.append(item)
    return HFDataset.from_list(trains), HFDataset.from_list(tests)
    # all_dataset = HFDataset.from_list(result)
    #
    # split = all_dataset.train_test_split(0.15, seed=1)
    # return split['train'], split['test']


train_dataset, eval_dataset = load_dataset_from_path()
print(
    f'total train size:{train_dataset.shape[0]},eval size:{eval_dataset.shape[0]}')


class SwinForRegression(nn.Module):
    def __init__(self, model_name="microsoft/swin-large-patch4-window12-384-in22k", num_labels=2):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(model_name)  # 只加载特征提取部分
        hidden_size = self.backbone.config.hidden_size  # Swin的最后一层特征维度
        self.regressor = nn.Linear(hidden_size, num_labels)  # 线性回归层

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values)  # 提取特征
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        logits = self.regressor(pooled_output)  # 线性映射到目标维度

        loss = None
        if labels is not None:
            # cri = MSELoss()
            # loss = cri(logits, labels)
            pred_sin, pred_cos = logits[:, 0], logits[:, 1]
            true_sin, true_cos = labels[:, 0], labels[:, 1]

            pred_angle = torch.atan2(pred_sin, pred_cos) * 180.0 / torch.pi
            true_angle = torch.atan2(true_sin, true_cos) * 180.0 / torch.pi
            angle_diff = torch.abs(pred_angle - true_angle)
            angle_diff = torch.min(angle_diff, 360.0 - angle_diff)
            loss = angle_diff.mean()

        return SwinImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            reshaped_hidden_states=outputs.reshaped_hidden_states,

        )


# 示例
model = SwinForRegression(model_name=MODEL_NAME_OR_PATH, num_labels=2)

processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME_OR_PATH)


class RegressionDataset(Dataset):
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

        sin_angle, cos_angle = angle_to_target(self.dataset[idx]['angle'])
        label = torch.Tensor((sin_angle, cos_angle))

        return {"image": image, "label": label}


# 创建数据集对象
train_dataset = RegressionDataset(train_dataset, processor)
eval_dataset = RegressionDataset(eval_dataset, processor)

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
    inputs["labels"] = torch.stack(labels)
    return inputs


def compute_metrics(eval_preds):
    pred_angles, label_angles = [], []
    acc, total = 0, 0
    labels = eval_preds.label_ids
    preds = eval_preds.predictions
    for (pred, label) in zip(preds, labels):
        pred_angle = pred_to_angle(sin_pred=pred[0], cos_pred=pred[1])
        label_angle = pred_to_angle(sin_pred=label[0], cos_pred=label[1])
        pred_angles.append(int(pred_angle))
        label_angles.append(int(label_angle))

        total += 1
        if abs(pred_angle - label_angle) <= 1:
            acc += 1
    print(f'ACC: {acc / (total + 1e-5)}')
    print(classification_report(y_true=label_angles, y_pred=pred_angles, ))
    return {
        "f1": f1_score(y_true=label_angles, y_pred=pred_angles, labels=list(range(0, 361)), average='macro'),
        "acc": acc / (total + 1e-5),
    }


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
