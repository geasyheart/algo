from typing import List, Tuple
from unittest import TestCase

from .trie import Trie
from .segtrie import SegmentationTrie


class TestTrie(TestCase):
    def test_trie(self):
        words = [
            "我爱中国",
            "我爱你中国",
            "爱你爱你"
        ]
        trie = Trie()
        for word in words:
            trie.add(word)

        for word in words:
            self.assertTrue(trie.contains(word))

        self.assertTrue(trie.has_prefix("爱"))
        self.assertTrue(trie.has_prefix("我"))
        self.assertFalse(trie.has_prefix("吗"))
        # 最终结尾的
        self.assertTrue(trie.search("...你"))
        self.assertTrue(trie.search("..你.."))  # 必须整个长度..
        print(trie.keys("爱"))

    def test_seq_trie(self):
        words: List[Tuple] = [
            ("我", "爱", "中国"),
            ("我", "爱", "你", "中国"),
            ("我", "爱", "你", "job"),
            ("我", "爱", "你", "job", "hi"),
            ("爱", "中国")
        ]

        seg_trie = SegmentationTrie()
        for seg_word in words:
            seg_trie.add(words=seg_word)
        # test contains
        for seg_word in words:
            self.assertTrue(seg_trie.contains(words=seg_word))
        self.assertFalse(seg_trie.contains(words=("我", "爱")))
        self.assertFalse(seg_trie.contains(words=("我", "爱", "你")))
        # test search
        for seg_word in words:
            self.assertTrue(seg_trie.search(seg_word))
        self.assertTrue(seg_trie.search(words=["我", "爱", "/"]))
        self.assertTrue(seg_trie.search(words=["我", "爱", "你", "/"]))
        self.assertFalse(seg_trie.search(words=["我", "爱", "U"]))
        # test keys
        for i in seg_trie.iter_keys(["我", "爱"]):
            print(i)
