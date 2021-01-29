# -*- coding: utf8 -*-

import os

import pandas as pd

from data_process import CSVSequence
from loader import build_albert_transformer
import tensorflow as tf
model, tokenizer = build_albert_transformer(albert_dir="/tmp/albert_base", num_labels=2)

# TODO: k-fold以及交叉生成新的数据集，增大样本数

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join(os.curdir, "data", 'train.csv'))
    train_data = CSVSequence(csv_df=train_df, token_dict=tokenizer.vocab, shuffle=True)

    valid_df = pd.read_csv(os.path.join(os.curdir, "data", 'dev.csv'))
    valid_data = CSVSequence(csv_df=valid_df, token_dict=tokenizer.vocab, shuffle=False)

    history = model.fit(
        train_data,
        validation_data=valid_data,
        epochs=20,
        batch_size=32,
        # use_multiprocessing=True,
        # workers=-1,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join("/tmp/", "similarity.model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                save_weights_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join("/tmp/", "similarity_log")
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                mode='max',
                verbose=1,
                patience=2
            )
        ],
    )
