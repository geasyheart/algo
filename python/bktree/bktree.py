from typing import List


def levenstein_distance(s1: str, s2: str) -> int:
    """Calculate minimal Levenstein distance between s1 and s2.

    Args:
        s1 (str): string
        s2 (str): string

    Returns:
        int: the minimal distance.
    """
    m, n = len(s1) + 1, len(s2) + 1

    # Initialize
    dp = [[0] * n for i in range(m)]
    dp[0][0] = 0
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + 1
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + 1

    for i in range(1, m):
        for j in range(1, n):
            if s1[i - 1] != s2[j - 1]:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
            else:
                dp[i][j] = dp[i - 1][j - 1]
    return dp[m - 1][n - 1]


class BurkhardKellerNode(object):
    """Node implementatation for BK-Tree. A BK-Tree node stores the information of current word, and its approximate words calculated by levenstein distance.

    Args:
        word (str): word of current node.
    """

    def __init__(self, word: str):
        self.word = word
        self.next = {}


class BurkhardKellerTree(object):
    """Implementataion of BK-Tree"""

    def __init__(self):
        self.root = None
        self.nodes = {}

    def __add(self, cur_node: BurkhardKellerNode, word: str):
        """Insert a word into current tree. If tree is empty, set this word to root.

        Args:
            word (str): word to be inserted.
        """
        if self.root is None:
            self.root = BurkhardKellerNode(word)
            return
        if word in self.nodes:
            return
        dist = levenstein_distance(word, cur_node.word)
        if dist not in cur_node.next:
            self.nodes[word] = cur_node.next[dist] = BurkhardKellerNode(word)
        else:
            self.__add(cur_node.next[dist], word)

    def add(self, word: str):
        """Insert a word into current tree. If tree is empty, set this word to root.

        Args:
            word (str): word to be inserted.
        """
        return self.__add(self.root, word)

    def __search_similar_word(self, cur_node: BurkhardKellerNode, s: str, threshold: int = 2) -> List[str]:
        res = []
        if cur_node is None:
            return res
        dist = levenstein_distance(cur_node.word, s)
        if dist <= threshold:
            res.append((cur_node.word, dist))
        start = max(dist - threshold, 1)
        while start < dist + threshold:
            tmp_res = self.__search_similar_word(cur_node.next.get(start, None), s)[:]
            res.extend(tmp_res)
            start += 1
        return res

    def search_similar_word(self, word: str) -> List[str]:
        """Search the most similar (minimal levenstain distance) word between `s`.

        Args:
            s (str): target word

        Returns:
            List[str]: similar words.
        """
        res = self.__search_similar_word(self.root, word)

        def max_prefix(s1: str, s2: str) -> int:
            res = 0
            length = min(len(s1), len(s2))
            for i in range(length):
                if s1[i] == s2[i]:
                    res += 1
                else:
                    break
            return res

        res.sort(key=lambda d: (d[1], -max_prefix(d[0], word)))
        return res
