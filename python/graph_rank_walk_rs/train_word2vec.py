# -*- coding: utf8 -*-
#
# 至于选择skip-gram还是cbow，这个也得尝试
import json
import os.path

import torch
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.functional import F

with open("unique_id.json", "r") as f:
    unique_id_map = json.loads(f.read())


class SkipGramDataset(Dataset):
    def __init__(self):
        self.data = []
        with open("data.jsonl", 'r') as f:
            for line in f:
                line = json.loads(line)
                # 按照中间的那个来
                w = unique_id_map[line[2]]
                context = [unique_id_map[v] for i, v in enumerate(line) if i != 2]

                self.data.extend([(w, c) for c in context])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples])
        targets = torch.tensor([ex[1] for ex in examples])
        return inputs, targets

    def to_dl(self, batch_size=32, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size):
        super(SkipGramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.output = nn.Linear(64, vocab_size)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        output = self.output(embed)
        return F.log_softmax(output, dim=-1)


nll_loss = nn.NLLLoss()
# 构建Skip-gram模型，并加载至device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SkipGramModel(len(unique_id_map))
model.to(device)
if os.path.exists("skipgram.pt"):
    print('loading finetune model...')
    model.load_state_dict(torch.load("skipgram.pt"))
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
data_loader = SkipGramDataset().to_dl(batch_size=4096, shuffle=True)
for epoch in range(100):
    total_loss = 0
    for batch in tqdm.tqdm(data_loader):
        inputs, targets = [x.to(device) for x in batch]
        optimizer.zero_grad()
        log_probs = model(inputs)
        loss = nll_loss(log_probs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(data_loader)
    print(f"Epoch:{epoch} Loss: {total_loss:.2f}")

torch.save(model.state_dict(), "skipgram.pt")
