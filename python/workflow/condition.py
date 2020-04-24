# coding=utf-8
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Condition(object):
    name: str = ""
    condition: Any = None
    parent_node: Any = None
