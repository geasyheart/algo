# -*- coding: utf8 -*-
#
from unittest import TestCase

import numpy as np

from hmm import HMM


class TestSomeCase(TestCase):
    def test_train(self):
        word_map = {
            "我": 0,
            "爱": 1,
            '你': 2,
            "呀": 3
        }
        nature_map = {
            'n': 0,
            'v': 1
        }
        samples = [
            (['我', '爱', '你'], ['n', 'v', 'n']),
            (['我', '爱', '你', '呀'], ['n', 'v', 'n', 'v']),
        ]
        train_set = []
        for words, natures in samples:
            word_ids = [word_map[i] for i in words]
            nature_ids = [nature_map[i] for i in natures]
            train_set.append((word_ids, nature_ids))

        hmm = HMM(nw=4, nt=2)
        hmm.train(
            trainset=train_set,
            alpha=0.3, file='/tmp/a.pkl')
        # 可以看到，A和B等于下面的
        # A: 状态转移概率矩阵，也就是词性
        A = np.array([[0., 2., 2.], [3., 0., 0.], [1., 1., 0.]])
        # B:从词到词性的发射概率矩阵
        B = np.array([[2., 0.], [0., 2.], [2., 0.], [0., 1.]])

        # 注意：如果self.strans或者self.etrans难理解，就把他想成<bos>和<eos>就行