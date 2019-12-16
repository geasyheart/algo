from typing import Dict, List, Iterable


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

    def add(self, word: str):
        """
        向Trie添加一个单词
        :param word:
        :return:
        """
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
        """
        判断是否包含这个word
        :param word:
        :return:
        """
        cur: Node = self.root
        for c in word:
            if cur.next.get(c) is None:
                return False
            cur = cur.next.get(c)
        return cur.is_word

    def has_prefix(self, prefix: str) -> bool:
        """
        通过前缀索引的方式进行匹配是否存在这个单词

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

    def keys(self, prefix: str) -> List[str]:
        """
        获取符合的字符串
        :param prefix:
        :return:
        """
        _all: List[str] = []
        # 此处这个方法和has_prefix一样
        cur: Node = self.root
        for c in prefix:
            if cur.next.get(c) is None:
                return _all
            cur = cur.next.get(c)
        self._get_keys(node=cur, prefix=prefix, result=_all)
        return _all

    def _get_keys(self, node: Node, prefix: str, result: List[str]):
        if node.is_word:
            result.append(prefix)

        for _char, next_node in node.next.items():
            next_word = f"{prefix}{_char}"
            if next_node.is_word:
                result.append(next_word)
            else:
                self._get_keys(node=next_node, prefix=next_word, result=result)

    def iter_keys(self, prefix: str) -> Iterable[str]:
        raise NotImplementedError
