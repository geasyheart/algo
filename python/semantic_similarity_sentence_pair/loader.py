# -*- coding: utf8 -*-
import os

import bert
from bert.tokenization.albert_tokenization import FullTokenizer
import tensorflow as tf


def build_albert_transformer(albert_dir: str, num_labels: int, max_seq_len: int = 128, tokenizer_only: bool = False):
    """
    more can see here:
        hanlp/layers/transformers/loader_tf.py
    :return:
    """
    tokenizer = FullTokenizer(vocab_file=os.path.join(albert_dir, 'vocab_chinese.txt'), do_lower_case=True)
    if tokenizer_only:
        return tokenizer

    bert_params = bert.albert_params(albert_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, name='albert')
    bert_ckpt_file = os.path.join(albert_dir, "model.ckpt-best")

    # model
    l_input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name='input_ids')
    l_token_type_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name='token_type_ids')

    # provide a custom token_type/segment id as a layer input
    output = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
    # pooler output
    output = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(output)

    output = tf.keras.layers.Dropout(bert_params.hidden_dropout, name='hidden_dropout')(output)
    logits = tf.keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=bert_params.initializer_range)
    )(output)

    model = tf.keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=logits)

    model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])

    bert.load_albert_weights(l_bert, bert_ckpt_file)
    model.summary()
    model.compile(
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(2e-5),
        metrics=['accuracy'],

    )

    return model, tokenizer
