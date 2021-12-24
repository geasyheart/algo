# -*- coding: utf8 -*-
#
# 设置所有参数
import random
from collections import Counter

import scipy
from scipy.spatial.distance import cosine
from torch import nn
from torch.functional import F
import numpy as np
import torch
from torch.nn import Module
from torch.utils import data as tud

# 对每个库设置随机种子
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

C = 3  # 窗口大小，注意这里的窗口大小为分别向左边和右边取C个词
K = 15  # 负采样样本数（噪声词）
epochs = 2
MAX_VOCAB_SIZE = 10000
EMBEDDING_SIZE = 100
batch_size = 32
lr = 0.2

# 读取并分割

with open("train.txt", "r") as f:
    text = f.read()

text = text.lower().split()  # 分割成单词列表
vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))  # 得到单词字典表，key是单词，value是次数
vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values()))  # 把不常用的单词都编码为"<UNK>"

# 构建词值对
word2idx = {word: i for i, word in enumerate(vocab_dict.keys())}
idx2word = {i: word for i, word in enumerate(vocab_dict.keys())}

# 计算和处理频率
word_counts = np.array(list(vocab_dict.values()), dtype=np.float32)
word_freqs = (word_counts / np.sum(word_counts)) ** (3. / 4.)  # 所有的频率为原来的 0.75 次方， 论文中


class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word2idx, word_freqs):
        '''
        :text: a list of words, all text from the training dataset
        :word2idx: the dictionary from word to index
        :word_freqs: the frequency of each word
        '''
        super(WordEmbeddingDataset, self).__init__()
        # 注意下面重写的方法
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]  # 把单词数字化表示。如果不在词典中，也表示为unk
        self.text_encoded = torch.LongTensor(self.text_encoded)  # nn.Embedding需要传入LongTensor类型
        self.word2idx = word2idx
        self.word_freqs = torch.Tensor(word_freqs)

    def __len__(self):
        return len(self.text_encoded)  # 返回所有单词的总数，即item的总数

    def __getitem__(self, idx):
        ''' 这个function用于返回：中心词（center_words），周围词（pos_words），负采样词（neg_words）
        :idx: 中心词索引
        '''
        center_words = self.text_encoded[idx]  # 取得中心词
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))  # 取出所有周围词索引，用于下面的tensor(list)操作
        pos_indices = [i % len(self.text_encoded) for i in
                       pos_indices]  # 为了避免索引越界，所以进行取余处理，如：设总词数为100，则[1]取余为[1]，而[101]取余为[1]
        pos_words = self.text_encoded[pos_indices]  # tensor(list)操作，取出所有周围词
        neg_indices = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
        neg_words = self.text_encoded[neg_indices]
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 实际上从这里就可以看出，这里用的是skip-gram方法，并且采用负采样（Negative Sampling）进行优化

        # while 循环是为了保证 neg_words中不能包含周围词
        # Angel Hair：实际上不需要这么处理，因为我们遇到的都是非常大的数据，会导致取到周围词的概率非常非常小，这里之所以这么做是因为本文和参考文所提供的数据太小，导致这个概率变大了，会影响模型
        while len(set(pos_indices) & set(neg_indices.numpy().tolist())) > 0:
            neg_indices = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)
            neg_words = self.text_encoded[neg_indices]

        return center_words, pos_words, neg_words


dataset = WordEmbeddingDataset(text, word2idx, word_freqs)
dataloader = tud.DataLoader(dataset, batch_size, shuffle=True)


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        '''return: loss, [batch_size]
        :input_labels: center words, [batch_size]
        :pos_labels: positive words, [batch_size, (C * 2)]
        :neg_labels：negative words, [batch_size, (C * 2 * K)]

        '''

        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]

        pos_embedding = self.out_embed(pos_labels)  # [batch_size, (window * 2), embed_size]

        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (window * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1]

        pos_dot = torch.bmm(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]

        neg_dot = -torch.bmm(neg_embedding, input_embedding)  # [batch_size, (window * 2 * K), 1]
        # 注意负号，参考公式可以知负样本的概率越小越好，所以位负号
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg  # 理论上应该除2，实际除不除一样

        return -loss  # 我们希望概率迭代越大越好，加一个负值，变成越小越好，使之可以正确迭代

    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()


model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train():
    for e in range(1):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            input_labels = input_labels.long()
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()  # .mean()默认不设置dim的时候，返回的是所有元素的平均值
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('epoch', e, 'iteration', i, loss.item())

    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))


def predict():
    model.load_state_dict(torch.load("embedding-100.th"))
    embedding_weights = model.input_embedding()

    def find_nearest(word):
        index = word2idx[word]
        embedding = embedding_weights[index]

        cos_dis = np.array([cosine(e, embedding) for e in embedding_weights])
        print(cos_dis.shape)
        return [idx2word[i] for i in cos_dis.argsort()[:10]]
    # ['陈平安', '她', '崔东山', '宁姚', '刘羡阳', '朱敛', '只是', '他', '裴钱', '李宝瓶']
    print(find_nearest("陈平安"))


if __name__ == '__main__':
    train()
    predict()
