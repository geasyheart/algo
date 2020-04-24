# coding=utf-8
from typing import Any


class WorkflowError(Exception):
    msg = None

    def __init__(self, msg: Any):
        self.msg = msg


class DummyNodeError(WorkflowError):
    msg = "Dummy Node"


class CircularDependencyError(WorkflowError):
    msg = "Circular Dependency Error"
