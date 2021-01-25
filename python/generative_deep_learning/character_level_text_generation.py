# -*- coding: utf8 -*-
import random
import sys

import numpy as np
import tensorflow as tf


def reweight_distribution(original_distribution, temperature=0.5):
    """
    original_distribution 是概率值组
    成的一维 Numpy 数组，这些概率值之
    和必须等于 1。 temperature 是一个
    因子，用于定量描述输出分布的熵

    返回原始分布重新加权后的结果。
    distribution 的求和可能不再等
    于 1，因此需要将它除以求和，以
    得到新的分布

    更高的温度得到的是熵更大的采样分布，会生成更加出人意料、更加无结构的生成数据，
    而更低的温度对应更小的随机性，以及更加可预测的生成数据

    :param original_distribution:
    :param temperature:
    :return:
    """
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)

# 中文单词分类太多.
with open("/home/yuzhang/nietzsche.txt", 'r') as f:
# with open("/home/yuzhang/jaychou_lyrics_s.txt", 'r', encoding='utf-8') as f:
    text = f.read().lower()
    print(len(text))

maxlen = 30
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)

x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


def get_model():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(128, input_shape=(maxlen, len(chars)), return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.33))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dropout(0.33))
    model.add(tf.keras.layers.Dense(len(chars), activation='softmax'))
    # optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-05)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


model = get_model()
for epoch in range(1, 200):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    # 随机选择一个文本种子
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + generated_text + '"')

    for temperature in (0.2, 0.5, 1.0, 1.2):
        print('------------temperature:', temperature)
        sys.stdout.write(generated_text)
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)

model.save_weights('/tmp/generate_txt.md.h5')
