import os

import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torch.nn import BCELoss
from torchvision import transforms
# 加载 SwinTransformer 模型和处理器
from transformers import (AutoFeatureExtractor, SwinModel, set_seed)
from transformers.models.swin.modeling_swin import SwinImageClassifierOutput

set_seed(1)

transform = transforms.Compose([
    transforms.RandomResizedCrop(384),
])


class SwinForClassify(nn.Module):
    def __init__(self, model_name="microsoft/swin-large-patch4-window12-384-in22k", num_labels=1):
        super().__init__()
        self.backbone = SwinModel.from_pretrained(model_name)  # 只加载特征提取部分
        hidden_size = self.backbone.config.hidden_size  # Swin的最后一层特征维度
        self.classify = nn.Linear(hidden_size, num_labels)  # 线性回归层

    @torch.no_grad()
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


if __name__ == "__main__":
    MODEL_NAME_OR_PATH = './swin-large-patch4-window12-384-in22k/'
    # 示例
    model = SwinForClassify(model_name=MODEL_NAME_OR_PATH, num_labels=1)
    state_dict = load_file("./final_model/model.safetensors")
    model.load_state_dict(state_dict)
    model.eval()

    processor = AutoFeatureExtractor.from_pretrained(MODEL_NAME_OR_PATH)

    test_image_dir = "../image-rotate/test_images"
    for image_name in os.listdir(test_image_dir):
        image_path = os.path.join(test_image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        input = processor([image], return_tensors='pt')
        output = model(**input).logits[0]
        print(image_name, output > 0.5, output)
