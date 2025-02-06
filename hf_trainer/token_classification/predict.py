import json
from typing import List

import torch
from seqeval.metrics.sequence_labeling import get_entities
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForTokenClassification, AutoTokenizer

checkpoint_path = './output/checkpoint-1540'
model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

with open('label_id_map.json', 'r') as f:
    LABEL_ID_MAP = json.load(f)

ID_LABEL_MAP = {v: k for k, v in LABEL_ID_MAP.items()}


def predict(texts: List[str]):
    input_ids = pad_sequence([
        torch.LongTensor(tokenizer.convert_tokens_to_ids([
            tokenizer.cls_token,
            *[_ for _ in text][:510],
            tokenizer.sep_token
        ]))
        for text in texts
    ], batch_first=True)
    token_type_ids = torch.zeros(size=input_ids.shape, dtype=torch.long)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    pred_label_ids = [_.tolist()[1:-1] for _ in
                      output.logits.argmax(-1)[attention_mask].split(attention_mask.sum(-1).tolist())]

    for text, label_ids in zip(texts, pred_label_ids):
        entities = get_entities([ID_LABEL_MAP[_] for _ in label_ids])
        for (label, si, ei) in entities:
            print(text, text[si:ei + 1], label)


if __name__ == '__main__':
    predict([
        '雁塔区电子城街道',
        '西安市雁塔区田家湾地铁口（A口）',
        '陕西省西安市高新区丈八街道科技路33号高新国际'
    ])
