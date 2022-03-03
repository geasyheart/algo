# coding=utf-8
import copy
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
        can_flow_nodes = node.lefts

        result = []
        for can_flow_node in can_flow_nodes:
            result.append(can_flow_node.to_json())

        return result

    def next_step(self, node_id: str, condition: Any = None):
        """
        如果为ProcessNode，则直接通过
        如果为DecisionNode，则根据输入(condition)判断condition条件，从而获得下一个流转node
        :param node_id:
        :param condition: 具体计算不在这里计算，此处condition直接为最终算的结果，然后根据设置的condition结果进行判断应该走哪个流程
        :return:
        """
        node: Optional[ProcessNode, DecisionNode] = self.local_graph.node_map[node_id]
        # ProcessNode只有一个流转方向，所以直接返回parents第一个
        if isinstance(node, ProcessNode):
            copy_parents = copy.deepcopy(node.parents)
            if len(copy_parents) != 0:
                return copy_parents.pop()
            else:
                return None
        # DecisionNode有多个流转方向，根据输入算出结果然后获取下一个流转node
        elif isinstance(node, DecisionNode):
            for key, _condition in node.condition.items():
                # 结果相等并且结果类型相同
                if _condition.condition == condition and type(_condition.condition) == type(condition):
                    return _condition.parent_node
        else:
            return

    def auto_flow(self):
        """
        自动流转
        :return:
        """
        # all_available_nodes = ["process-1",,, "process-6"]
        # for cur_node in all_available_nodes:
            # condition = read_result_from_caled_result(cur_node_condition)
            # next_step = self.next_step(cur_node, condition)
            # yield next_step

    def trigger(self, node):
        """
        基于触发器的方式进行自动流转
        :return:
        """
        action = node.get_todo_action()
        action.apply_async(
            args=None,
            kwargs=None,
            eta=None,
            countdown=3,
            priority=None
        )
