# coding=utf-8

from typing import Dict, Iterable, Iterator, Tuple, Optional, List


class Node(object):
    def __init__(self, is_word: bool = False):
        """

        :param is_word: is a word
        """
        self.is_word = is_word
        self.next: Dict[str, Node] = dict()

    def words(self, prefix: Tuple[str]) -> Iterator:
        if self.is_word:
            yield prefix

        for word, child in self.next.items():
            yield from child.words((*prefix, word))


class SegmentationTrie(object):
    def __init__(self, delimiter: str = "/"):
        self.root: Node = Node()
        self.size: int = 0
        self.delimiter: str = delimiter

    def add(self, words):
        """
        例如:
            "我爱中国" 分词变成了 ["我", "爱", "中国"]
        那么此处:
            add(words=("我", "爱", "中国"))
        :param words: 
        :return: 
        """
        cur: Node = self.root
        for word in words:
            cur = cur.next.setdefault(word, Node())
        if not cur.is_word:
            cur.is_word = True
            self.size += 1

    def contains(self, words) -> bool:
        """
        判断是否包含此words
        例如:
            contains(words=["我", "爱"])
        :param words: 
        :return: 
        """
        cur: Node = self.root
        for word in words:
            cur = cur.next.get(word)
            if cur is None:
                return False
        return cur.is_word

    def has_prefix(self, prefix) -> bool:
        """
        通过前缀索引的方式进行匹配是否存在这个单词
        :param prefix: 
        :return: 
        """
        cur: Node = self.root
        for word in prefix:
            cur = cur.next.get(word)
            if cur is None:
                return False
        return True

    def search(self, words) -> bool:
        """
        通过模糊匹配的方式进行判断是否存在
        :param words:
        :return:
        """
        return self._search(
            node=self.root,
            words=words,
            index=0
        )

    def _search(self, node: Node, words, index: int) -> bool:
        if index == len(words):
            return node.is_word
        word = words[index]
        if word != self.delimiter:
            node = node.next.get(word)
            if node is None:
                return False
            return self._search(
                node=node,
                words=words,
                index=index + 1
            )
        else:
            for word in node.next.keys():
                if self._search(node.next.get(word), words, index + 1):
                    return True
            return False

    def iter_keys(self, prefix):
        """

        :param prefix:
        :return:
        """
        cur: Node = self.root
        # verify
        for word in prefix:
            cur = cur.next.get(word)
            if cur is None:
                yield []
        yield from cur.words(prefix=prefix)
