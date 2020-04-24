# coding=utf-8
from typing import Optional, List, Dict, Any

from graph import Graph
from node import ProcessNode, DecisionNode


class WorkflowEngine(object):
    def __init__(self, filename):
        with open(filename) as f:
            xml_text = f.read()

        self.local_graph: Graph = self.build_graph(xml_text=xml_text)

    def build_graph(self, xml_text: str) -> Graph:
        return Graph(xml_text).build_graph()

    def current_available_transition(self, node_id: str) -> List[Dict[str, Any]]:
        """
        获取能够流转的方向，包含条件判断node

        node_id为 xml 中的唯一id
        :return:
        """
        node: Optional[ProcessNode, DecisionNode] = self.local_graph.node_map[node_id]
        can_flow_nodes = node.parents

        result = []
        for can_flow_node in can_flow_nodes:
            result.append(can_flow_node.to_json())

        return result


