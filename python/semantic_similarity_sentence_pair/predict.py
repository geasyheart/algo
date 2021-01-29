# -*- coding: utf8 -*-
from typing import List

import numpy as np

from custom_tokenizer import Tokenizer
from loader import build_albert_transformer
import pandas as pd
import os
from data_process import CSVSequence

model, tokenizer = build_albert_transformer(albert_dir="/tmp/albert_base", num_labels=2)

t = Tokenizer(tokenizer.vocab)
model.load_weights('/tmp/similarity.model.h5')


def predict(pairs) -> List[bool]:
    """
    true: 相似

    :param pairs:
    :return:
    """

    input_ids, token_type_ids = [], []
    for sentence1, sentence2 in pairs:
        input_id, token_type_id = t.encode(first=sentence1, second=sentence2, max_len=128)
        input_ids.append(input_id)
        token_type_ids.append(token_type_id)

    p = model.predict(
        [
            np.array(input_ids, dtype=np.int32),
            np.array(token_type_ids, dtype=np.int32)
        ]
    )
    return (p > 0.5).flatten().tolist()


if __name__ == '__main__':
    p = predict([
        ('肝癌兼肺癌晚期能活多久？', '肝癌兼肺癌晚期还有多少寿命？'),
        ("剧烈运动后咯血,是怎么了?", "剧烈运动后咯血，应该怎么处理？")
    ])
    print(p)
