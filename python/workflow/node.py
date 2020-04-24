# coding=utf-8
from typing import Set, Type, Dict, Any

from condition import Condition


class Node(object):
    def __init__(self, key=""):
        self.key = key
        self.attrib: Dict[str, Any] = {}

        self.parents: Set[Type[Node]] = set()
        self.childrens: Set[Type[Node]] = set()

    def set_attrib(self, attrib: Dict[str, Any]):
        self.attrib = attrib

    def add_parent(self, parent: Type['Node']):
        self.parents.add(parent)

    def add_children(self, children: Type['Node']):
        self.childrens.add(children)

    def current_available_transition(self):
        for parent in self.parents:
            pass

    def __repr__(self):
        return f"<{self.__class__.__name__}>: {self.key}"


class ProcessNode(Node):
    def __init__(self, key=""):
        super(ProcessNode, self).__init__(key=key)

    def to_json(self):
        return {
            "key": self.key,
            "attrib": self.attrib
        }


class DecisionNode(Node):
    def __init__(self, key=""):
        super(DecisionNode, self).__init__(key=key)
        # # 但是此处有可能非两个条件，而是满足一定条件后的流转.
        # 简单来说，此处会有多条流转判断边
        # 条件名字: 条件满足条件(有可能为函数，有可能就是一个简单的bool值)
        self.condition: Dict[str, Condition] = {}

    def add_condition(self, name: str, condition: Condition):
        # self.auto_pass: Optional[None, bool] = None
        # self.auto_reject: Optional[None, bool] = None
        # self.condition: str = ""
        self.condition.update({
            name: condition
        })

    def to_json(self):
        return {
            "key": self.key,
            "attrib": self.attrib,
            "condition": {
                key: condition.to_json()
                for key, condition in self.condition.items()
            }
        }


class DummyNode(Node):
    def raise_error(self):
        raise ValueError("Dummy Node")
