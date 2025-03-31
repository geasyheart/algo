import os
import pprint
import random
import re
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import tqdm
from PIL import Image
from datasets import Dataset as HFDataset
from safetensors.torch import load_file
from sklearn.metrics import classification_report, f1_score
from torch.nn import MSELoss
from torch.utils.data import Dataset
from torchvision import transforms
# 加载 SwinTransformer 模型和处理器
from transformers import (AutoFeatureExtractor, SwinModel, Trainer,
                          TrainingArguments, set_seed)
from transformers.models.swin.modeling_swin import SwinImageClassifierOutput

set_seed(1)


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


class SwinForRegression(nn.Module):
    def __init__(self, model_name="microsoft/swin-large-patch4-window12-384-in22k", num_labels=2):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(model_name)  # 只加载特征提取部分
        hidden_size = self.backbone.config.hidden_size  # Swin的最后一层特征维度
        self.regressor = nn.Linear(hidden_size, num_labels)  # 线性回归层

    @torch.no_grad()
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


if __name__ == "__main__":
    device = torch.device('cuda:0')
    MODEL_NAME_OR_PATH = './swin-large-patch4-window12-384-in22k'
    model = SwinForRegression(model_name=MODEL_NAME_OR_PATH, num_labels=2).to(device)
    state_dict = load_file("./final_model/model.safetensors")
    model.load_state_dict(state_dict)
    model.eval()
    processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME_OR_PATH)

    right, total = 0, 0
    test_image_dirs = ['./test_images', './人民日报倾斜']
    for test_image_dir in test_image_dirs:
        for image_name in os.listdir(test_image_dir):
            image_path = os.path.join(test_image_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            input = processor([image], return_tensors='pt').to(device)
            output = model(**input)
            sin_pred, cos_pred = output.logits[0].cpu().tolist()
            pred_angle = pred_to_angle(sin_pred, cos_pred)
            p = re.search(r"angle\-(\d+)", image_name)
            if p:
                total += 1
                right_angle = int(p.group(1))
                angle_diff = abs(right_angle - pred_angle)
                angle_diff = min(angle_diff, 360 - angle_diff)
                if angle_diff <= 1:
                    right += 1

                print(f'{right=}, {total=}, {image_name}, {pred_angle}')
