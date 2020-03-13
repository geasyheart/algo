# coding=utf-8
from dataclasses import dataclass, asdict
from typing import Optional, Union


@dataclass
class Table(object):
    pk: int
    name: str
    parent: Union[int, None]

    def to_dict(self):
        return asdict(self)
