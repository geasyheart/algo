# -*- coding: utf8 -*-

from jieba import posseg

from textrank import TextRank


def cut_with_hanlp(sentence):
    result = posseg.lcut(sentence)
    return [(i.word, i.flag) for i in result]

text = """
中国博物馆故宫天安门
"""


def test_with_hanlp_textrank():
    text_rank = TextRank()
    result = text_rank.textrank(
        cut_with_hanlp(text),
        topK=10,
        # withWeight=True,
        # withFlag=True,
    )
    print(result)


if __name__ == '__main__':
    test_with_hanlp_textrank()
