# import os
# # os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['http_proxy'] = 'http://192.168.0.122:1080'
# os.environ['https_proxy'] = 'http://192.168.0.122:1080'
#
# from datasets import load_dataset
#
# dataset = load_dataset('lmms-lab/VQAv2', split='validation[:100]')
# print()

from datasets import load_dataset
from tqdm import tqdm
from transformers import ViltConfig

dataset = load_dataset('parquet', data_files='/home/yuzhang/桌面/validation-00000-of-00068.parquet')['train']

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
print()


def get_score(count: int) -> float:
    return min(1.0, count / 3)


annotations = []
for i in range(10):
    sample = dataset[i]
    answer_count = {}
    for answer in sample['answers']:
        answer_ = answer["answer"]
        answer_count[answer_] = answer_count.get(answer_, 0) + 1

    labels, scores = [], []
    for answer in answer_count:
        if answer not in list(config.label2id.keys()):
            continue

        labels.append(config.label2id[answer])
        score = get_score(answer_count[answer])
        scores.append(score)
    sample['labels'] = labels
    sample['scores'] = scores

    annotations.append(sample)

import torch


class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # get image + text
        # annotation = self.annotations[idx]
        # questions = self.questions[idx]
        # image = Image.open(id_to_filename[annotation['image_id']])
        item = self.data[idx]

        text = item['question']
        image = item['image']
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        # add labels
        labels = item['labels']
        scores = item['scores']
        # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
        targets = torch.zeros(len(config.id2label))
        for label, score in zip(labels, scores):
            targets[label] = score
        encoding["labels"] = targets

        return encoding


from transformers import ViltProcessor

MODEL_NAME_OR_PATH = '/home/yuzhang/windows_share/python-packages/pretrained/vilt-b32-mlm/'
processor = ViltProcessor.from_pretrained(MODEL_NAME_OR_PATH)

dataset = VQADataset(data=annotations,
                     processor=processor)
list(dataset)
from transformers import ViltForQuestionAnswering

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViltForQuestionAnswering.from_pretrained(MODEL_NAME_OR_PATH,
                                                 id2label=config.id2label,
                                                 label2id=config.label2id)
model.to(device)

from torch.utils.data import DataLoader


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # create padded pixel values and corresponding pixel mask
    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels)

    return batch


train_dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)

batch = next(iter(train_dataloader))

for k, v in batch.items():
    print(k, v.shape)

from PIL import Image
import numpy as np

image_mean = processor.image_processor.image_mean
image_std = processor.image_processor.image_std

batch_idx = 1

unnormalized_image = (batch["pixel_values"][batch_idx].numpy() * np.array(image_mean)[:, None, None]) + np.array(
    image_std)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
image = Image.fromarray(unnormalized_image)
print(processor.decode(batch["input_ids"][batch_idx]))
labels = torch.nonzero(batch['labels'][batch_idx]).squeeze().tolist()
print([config.id2label[label] for label in labels])

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(50):  # loop over the dataset multiple times
    print(f"Epoch: {epoch}")
    for batch in tqdm(train_dataloader):
        # get the inputs;
        batch = {k: v.to(device) for k, v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(**batch)
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        optimizer.step()

print()
