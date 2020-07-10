# -*- coding: utf8 -*-

from pyhanlp import HanLP
from .textrank import TextRank


def cut_with_hanlp(sentence):
    result = []
    for p in HanLP.segment(sentence):
        result.append((p.word, str(p.nature)))
    return result


text = """
中国博物馆，故宫，受持续强降雨影响，河南省商丘市永城段昨天出现河水倒灌，周边多个村庄受淹。当地出动多台大排量排水车，强排内涝积水。从5日开始，广东清远、韶关、广州等地遭遇强降雨。广东省防总昨天启动四级救灾应急响应，截至目前共转移安置群众43100多人。
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
