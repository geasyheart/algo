from unittest import TestCase
from hashlib import md5
import marisa_trie


class TestImplTrie(TestCase):

    def test_trie(self):
        # 都不需要分词
        to_one_line = ['我爱你中国', '我爱到家', '到家美好']
        trie = marisa_trie.Trie(to_one_line)
        print(trie.keys("我爱"))


