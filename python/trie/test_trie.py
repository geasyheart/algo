from unittest import TestCase

from .trie import Trie


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
