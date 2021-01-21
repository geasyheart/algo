# -*- coding: utf8 -*-
import os
from typing import List
# bert-for-tf2
import bert
import numpy as np
import tensorflow as tf
# from bert.loader_albert import albert_models_google # get address
from bert.tokenization.albert_tokenization import FullTokenizer
from tensorflow import keras

# 关于hanlp func you can see git: d501bdb295ae7025e0975c1187caabcf12ed42b1
from hanlp.datasets.cws.ctb6 import CTB6_CWS_DEV, CTB6_CWS_TRAIN, CTB6_CWS_TEST
from hanlp.utils.io_util import get_resource
from hanlp.utils.span_util import bmes_of, bmes_to_words

max_seq_len = 128
batch_size = 16
LABEL_MAP = {'B': 1, 'M': 2, 'E': 3, 'S': 4}


def build_transformer(bert_dir: str = '/tmp/albert_base'):
    """
    more can see here:
        hanlp/layers/transformers/loader_tf.py

    :param bert_dir:
    :return:
    """
    tokenizer = FullTokenizer(vocab_file=os.path.join(bert_dir, 'vocab_chinese.txt'), do_lower_case=True)

    bert_params = bert.albert_params(bert_dir)
    l_bert = bert.BertModelLayer.from_params(bert_params, name='albert')
    bert_ckpt_file = os.path.join(bert_dir, "model.ckpt-best")

    # model
    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name='input_ids')
    l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name='token_type_ids')

    # provide a custom token_type/segment id as a layer input
    output = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]

    output = tf.keras.layers.Dropout(bert_params.hidden_dropout, name='hidden_dropout')(output)
    logits = tf.keras.layers.Dense(len(LABEL_MAP) + 1, activation='softmax')(output)

    model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=logits)

    model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])

    bert.load_albert_weights(l_bert, bert_ckpt_file)
    model.summary()

    model.compile(
        # recommended learning rate for Adam 5e-5, 3e-5, 2e-5
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5, ),
        # SparseCategoricalCrossentropy 和 CategoricalCrossentropy的简单区别在于
        # SparseCategoricalCrossentropy可以不用对标签做one-hot编码
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
    )

    return model, tokenizer


model, tokenizer = build_transformer()


def file_to_inputs(file: str):
    filepath = get_resource(file)
    with open(filepath, encoding='utf-8') as src:
        for line in src:
            sentence = line.strip()
            if not sentence:
                continue
            yield sentence


def convert_example_to_feature(line: str):
    _tokens, labels = bmes_of(line, True)
    tokens = tokenizer.tokenize(line.replace(" ", ''))
    tokens = ['[CLS]'] + tokens[:max_seq_len - 2] + ['[SEP]']
    padding_len = max_seq_len - len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens=tokens)

    bert_input = {
        "input_ids": input_ids + [0] * padding_len,
        "token_type_ids": [0] * max_seq_len
    }
    label_ids = [LABEL_MAP[_] for _ in labels]
    label_ids = label_ids[:max_seq_len - 2]
    padding_len = max_seq_len - len(label_ids)

    return bert_input, label_ids + [0] * padding_len


def map_example_to_dict(input_ids, token_type_ids, label):
    return {
               "input_ids": input_ids,
               "token_type_ids": token_type_ids,
           }, label


def encode_files(file):
    input_ids_list = []
    token_type_ids_list = []
    label_list = []

    for line in file_to_inputs(file=file):
        bert_input, labels = convert_example_to_feature(line=line)

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        label_list.append(labels)
    return [np.array(input_ids_list, dtype=np.int32), np.array(token_type_ids_list, dtype=np.int32)], np.array(
        label_list, dtype=np.int32)
    # ds = tf.data.Dataset.from_tensor_slices((input_ids_list, token_type_ids_list, label_list)).map(map_example_to_dict)
    # return ds.shuffle(1000).batch(batch_size)


def train():
    trn_data, trn_label = encode_files(CTB6_CWS_TRAIN)
    dev_data, dev_label = encode_files(CTB6_CWS_DEV)
    model.fit(
        trn_data,
        trn_label,
        epochs=2,
        batch_size=batch_size,
        validation_data=(dev_data, dev_label),
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join("/tmp/", "cws.model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                save_weights_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join("/tmp/", "cws_monitor_log")
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                mode='max',
                verbose=1,
                patience=1
            )
        ],

    )

# FIXME: 可以改一下的噻。
loaded = False


def predict(sentences: List[str]):
    global loaded
    if not loaded:
        model.load_weights("/tmp/cws.model.h5")
        loaded = True

    input_ids_list, token_type_ids_list = [], []
    for sentence in sentences:
        bert_input, _ = convert_example_to_feature(sentence)
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])

    sentences_predict_result = model.predict(
        [np.array(input_ids_list, dtype=np.int32), np.array(token_type_ids_list, dtype=np.int32)])

    index_label_map = {v: k for k, v in LABEL_MAP.items()}
    result = []
    for index, sentence_predict_result in enumerate(sentences_predict_result):
        sentence_label = []
        for char_prob in sentence_predict_result:
            char_index = np.argmax(char_prob)
            sentence_label.append(index_label_map.get(char_index, '[PAD]'))

        result.append(bmes_to_words(chars=list(sentences[index]), tags=sentence_label))
    return result


def evaluate():
    global loaded
    if not loaded:
        model.load_weights("/tmp/cws.model.h5")
        loaded = True
    test_data, test_label = encode_files(CTB6_CWS_TEST)
    result = model.evaluate(test_data, test_label)
    print(result)


if __name__ == '__main__':
    # train()
    # print(predict(['我爱你中国', '中华人民共和国', '今天中午看了一部电影']))
    evaluate()
