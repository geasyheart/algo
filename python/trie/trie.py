from typing import Dict


class Node(object):
    def __init__(self, is_word: bool = False):
        self.is_word = is_word
        self.next: Dict[str, Node] = dict()


class Trie(object):
    def __init__(self):
        self.root: Node = Node()
        self.size: int = 0

    def get_size(self) -> int:
        return self.size

    # 向Trie添加一个单词
    def add(self, word: str):
        cur: Node = self.root
        for c in word:
            if cur.next.get(c) is None:
                cur.next.update({c: Node()})
            cur = cur.next.get(c)
        if not cur.is_word:
            cur.is_word = True
            self.size += 1

    # 查询
    def contains(self, word: str) -> bool:
        cur: Node = self.root
        for c in word:
            if cur.next.get(c) is None:
                return False
            cur = cur.next.get(c)
        return cur.is_word

    # 通过前缀索引的方式进行匹配是否存在这个单词
    def has_prefix(self, prefix: str) -> bool:
        """
        trie: 我爱中国
        prefix: 我爱
        :param prefix: 
        :return: 
        """
        cur: Node = self.root
        for c in prefix:
            if cur.next.get(c) is None:
                return False
            cur = cur.next.get(c)
        # 到结尾了仍存在，那么则返回true
        return True

    def search(self, word: str) -> bool:
        """
        通过模糊匹配的方式进行判断是否存在

        例如搜索 a..b，此时的.代表任意字符

        :param word:
        :return:
        """
        return self._match(
            node=self.root,
            word=word,
            index=0
        )

    def _match(self, node: Node, word: str, index: int) -> bool:
        if index == len(word):
            return node.is_word
        c = word[index]
        if c != ".":
            node = node.next.get(c)
            if node is None:
                return False
            return self._match(
                node=node,
                word=word,
                index=index + 1
            )
        else:
            for next_char in node.next.keys():
                if self._match(node.next.get(next_char), word, index + 1):
                    return True
            return False

    def delete(self, word: str):
        """
        情况1: single tree(继续往上找，然后删除此word)
        情况2: 最后一个被其他引用(将最后一个char的is_word标志成false，如果不是true,则忽略)
        :param word:
        :return:
        """
        pass