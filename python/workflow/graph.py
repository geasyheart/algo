# coding=utf-8

from typing import Dict, Optional

from lxml import objectify
from lxml.html import Element

from condition import Condition
from exception import CircularDependencyError
from node import ProcessNode, DecisionNode, DummyNode


class Graph(object):
    def __init__(self, xml_text: str):

        self.node_map: Dict[str, Optional[ProcessNode, DecisionNode]] = {}
        self.node_info: Dict[str, Element] = {}  # 保存element原始信息

        self.xml_text = xml_text
        self.workflow_obj: Element = objectify.fromstring(self.xml_text)

    def build_graph(self):
        # 1. add all node
        self.add_node()
        # 2. 构建父子关系
        self.add_dependencies()
        # 3. validate consistency
        self.validate_consistency()
        # 4. 由于流程图本身会出现循环的情况，所以此处忽略
        # self.ensure_not_cyclic()
        return self

    def add_node(self):
        for node_ele in self.workflow_obj.iterchildren():
            node_ele_id: str = node_ele.attrib["id"]
            self.node_info[node_ele_id] = node_ele

            if node_ele.tag.lower() == "processnode":
                pn = ProcessNode(key=node_ele_id)
                pn.set_attrib(node_ele.attrib)
                self.node_map[node_ele_id] = pn
            elif node_ele.tag.lower() == "decisionnode":
                dn = DecisionNode(key=node_ele_id)
                dn.set_attrib(node_ele.attrib)
                self.node_map[node_ele_id] = dn
            else:
                # 添加dummy node.
                self.node_info[node_ele_id] = DummyNode(key=node_ele_id)

    def add_dependencies(self):
        for node_ele in self.workflow_obj.iterchildren():
            for transition_edge_ele in node_ele.iterchildren():
                self.node_map[node_ele.attrib['id']].add_left(
                    left=self.node_map[transition_edge_ele.attrib['to']])
                self.node_map[transition_edge_ele.attrib['to']].add_right(
                    children=self.node_map[node_ele.attrib['id']])

        # 条件node特殊处理
        for node_ele in self.workflow_obj.iterchildren():
            if node_ele.tag.lower() == "decisionnode":
                for transition_edge_ele in node_ele.iterchildren():
                    if transition_edge_ele.tag.lower() == "true-transition":
                        condition = Condition(
                            key=transition_edge_ele.tag.lower(),
                            condition=True,
                            parent_node=self.node_map[transition_edge_ele.attrib['to']]
                        )
                    elif transition_edge_ele.tag.lower() == "false-transition":
                        condition = Condition(
                            key=transition_edge_ele.tag.lower(),
                            condition=False,
                            parent_node=self.node_map[transition_edge_ele.attrib['to']]
                        )
                        # self.node_map[node_ele.attrib["id"]].set_false_condition(
                        #     self.node_map[transition_edge_ele.attrib['to']])
                    else:
                        # TODO: here
                        condition = Condition()
                    self.node_map[node_ele.attrib['id']].add_condition(transition_edge_ele.tag.lower(), condition)

    def validate_consistency(self):
        [n.raise_error() for n in self.node_map.values() if isinstance(n, DummyNode)]

    def ensure_not_cyclic(self):
        # Algo from GvR:
        # https://neopythonic.blogspot.com/2009/01/detecting-cycles-in-directed-graph.html

        todo = set(self.node_map)
        while todo:
            node = todo.pop()
            stack = [node]
            while stack:
                top = stack[-1]
                for child in self.node_map[top].childrens:
                    # Use child.key instead of child to speed up the frequent
                    # hashing.
                    node = child.key
                    if node in stack:
                        cycle = stack[stack.index(node):]
                        raise CircularDependencyError(cycle)
                    if node in todo:
                        stack.append(node)
                        todo.remove(node)
                        break
                else:
                    node = stack.pop()
