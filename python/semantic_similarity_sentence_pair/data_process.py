# -*- coding: utf8 -*-
import os

import numpy as np

import tensorflow as tf

from custom_tokenizer import Tokenizer


class CSVSequence(tf.keras.utils.Sequence):
    def __init__(self, csv_df, token_dict, batch_size=32, shuffle=True, max_length=128):
        self.csv_df = csv_df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_length = max_length
        self.indexes = np.arange(len(self.csv_df))

        # encoder.
        self.tokenizer = Tokenizer(token_dict=token_dict)

    def __len__(self):
        return len(self.csv_df) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_df = self.csv_df.iloc[indexes]
        labels = []
        input_ids, token_type_ids = [], []

        for index, row in batch_df.iterrows():
            input_id, token_type_id = self.tokenizer.encode(
                first=row['query1'],
                second=row['query2'],
                max_len=self.max_length
            )
            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            labels.append(row['label'])
        input_ids = np.array(input_ids, dtype=np.int32)
        token_type_ids = np.array(token_type_ids, dtype=np.int32)
        labels = np.array(labels, dtype=np.int32)

        return [input_ids, token_type_ids], labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.RandomState(333).shuffle(self.indexes)
