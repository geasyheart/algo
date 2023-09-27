# -*- coding: utf8 -*-
#
from typing import List

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def cal_blue(reference: List[List[str]], candidate: List[str]):
    """
    :param reference: 预测答案
    :param candidate: 标准答案
    """
    sm = SmoothingFunction()
    gram1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=sm.method1)
    gram2 = sentence_bleu(reference, candidate, weights=(0, 1, 0, 0), smoothing_function=sm.method1)
    gram3 = sentence_bleu(reference, candidate, weights=(0, 0, 1, 0), smoothing_function=sm.method1)
    gram4 = sentence_bleu(reference, candidate, weights=(0, 0, 0, 1), smoothing_function=sm.method1)

    return gram1, gram2, gram3, gram4, sentence_bleu(reference, candidate, smoothing_function=sm.method1)


if __name__ == '__main__':
    print(cal_blue([['a', 'cat', 'is', 'on', 'the', 'table']], ['there', 'is', 'a', 'cat', 'on', 'the', 'table']))
    print(cal_blue([['this', 'is', 'a', 'test']], ['this', 'is', 'a', 'test']))
