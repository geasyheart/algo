# coding=utf-8
from typing import Any


class Condition(object):
    def __init__(
            self,
            key: str = "",
            condition: Any = None,
            parent_node: Any = None
    ):
        self.key = key
        self.condition = condition
        self.parent_node = parent_node

    def to_json(self):
        return {
            "key": self.key,
            "condition": self.condition,  # 这里能不能展示？
            "parent_node": self.parent_node.to_json()
        }
