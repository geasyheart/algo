import logging
from logging import StreamHandler
from typing import Optional, List

logger = logging.getLogger(__name__)
logger.addHandler(StreamHandler())
logger.setLevel(logging.INFO)


class Node(object):
    def __init__(
            self,
            e: int = None,
            left_node: 'Node' = None,
            right_node: 'Node' = None
    ):
        self.e = e
        self.left_node = left_node
        self.right_node = right_node

    def __repr__(self):
        return f"<Node>: {self.e}"


class BST(object):
    """
    binary search tree. 二分搜索树
    """

    def __init__(self):
        self.root: Optional[Node, None] = None

    def add(self, e: int):
        if self.root is None:
            self.root = Node(e)
        else:
            self._add(self.root, e)

    def _add(self, node, e: int):
        logger.info(f"add element:{e}")
        if node.e < e and node.right_node is None:
            node.right_node = Node(e)
        elif node.e < e and node.right_node is not None:
            return self._add(node.right_node, e)
        elif node.e > e and node.left_node is None:
            node.left_node = Node(e)
        elif node.e > e and node.left_node is not None:
            return self._add(node.left_node, e)
        else:
            return

    def contains(self, e: int) -> bool:
        return self._contains(self.root, e)

    def _contains(self, node: Node, e: int) -> bool:
        if node is None:
            return False
        elif node.e == e:
            return True
        elif node.e < e:
            return self._contains(node.right_node, e)
        else:
            return self._contains(node.left_node, e)

    def pre_order(self):
        self._pre_order(self.root)

    def _pre_order(self, node: Node):
        if node is None:
            return
        # 前序遍历打印
        logger.info(node.e)
        self._pre_order(node.left_node)
        self._pre_order(node.right_node)

    def in_order(self):
        self._in_order(self.root)

    def _in_order(self, node: Node):
        if node is None:
            return
        self._in_order(node.left_node)
        logger.info(node.e)
        self._in_order(node.right_node)

    def post_order(self):
        self._post_order(self.root)

    def _post_order(self, node: Node):
        if node is None:
            return
        self._post_order(node.left_node)
        self._post_order(node.right_node)
        logger.info(node.e)

    def level_order(self):
        self._level_order([self.root])

    def _level_order(self, nodes: List[Node]):
        if not nodes:
            return
        next_level = []

        cur_level = []
        for node in nodes:
            cur_level.append(node.e)
            if node.left_node:
                next_level.append(node.left_node)
            if node.right_node:
                next_level.append(node.right_node)
        logger.info(cur_level)
        self._level_order(nodes=next_level)

    @property
    def minimum(self):
        return self._minimum(self.root)

    def _minimum(self, node:Node):
        if not node:
            return
        if node.left_node:
            return self._minimum(node.left_node)
        else:
            return node

    @property
    def maximum(self):
        return self._maximum(self.root)

    def _maximum(self, node: Node):
        if not node:
            return
        if node.right_node:
            return self._maximum(node.right_node)
        return node

    def del_min(self) -> Node:
        minimum = self.minimum
        self._del_min(self.root)
        return minimum

    # TODO: here
    def _del_min(self, node: Node):
        if node.left_node is None:
            right_node = node.right_node
            node.right_node = None
            return right_node
        node.left_node = self._del_min(node.left_node)
        return node

    def del_max(self):
        maximum = self.maximum
        self._del_max(self.root)
        return maximum

    def _del_max(self, node: Node):
        if node.right_node is None:
            left_node = node.left_node
            node.left_node = None
            return left_node
        node.right_node = self._del_max(node.right_node)
        return node

    def del_node(self, node: Node):
        raise NotImplementedError
