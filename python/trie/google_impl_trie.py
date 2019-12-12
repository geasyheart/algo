import pygtrie
from  unittest import TestCase


class TestImplTrie(TestCase):
    def test_char_trie(self):
        t = pygtrie.StringTrie()
        t['foo/hehe'] = 'Foo'
        t['foo/bar/baz'] = 'Baz'

        # t['我/爱/你/中国'] = '中国'
        # t['我/爱/你/天安门'] = "天安门"
        # rs = t.longest_prefix("我/爱/你/哈哈哈")
        # print(rs)

        t['wo/ai/ni/zhongguo'] = 'foo'
        t['wo/ai/ni/tiananmen'] = 'baz'

        rs = t.longest_prefix('foo')
        print(rs)

        rs = t.longest_prefix("wo/")
        print(rs)

    def test_trie(self):
        trie = pygtrie.StringTrie()
        words = ["我", "爱", "你", "中国"]
        for word in words:
            trie.update({word})