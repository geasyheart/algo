# coding=utf-8
from pprint import pprint
from unittest import TestCase

from engine import WorkflowEngine
from graph import Graph
from node import DecisionNode, ProcessNode


class TestWorkFlow(TestCase):
    def get_xml_text(self):
        with open("./sample2.xml") as f:
            xml_text = f.read()
        return xml_text

    def test_build_graph(self):
        xml_text = self.get_xml_text()
        graph = Graph(xml_text=xml_text)
        graph.build_graph()

        for id_, node in graph.node_map.items():
            if id_ == "process-1":
                self.assertEqual(node.key, "process-1")
                self.assertEqual(node.attrib['name'], "流程节点1")

                self.assertEqual(len(node.childrens), 0)
                self.assertEqual(len(node.parents), 1)
                self.assertEqual([i.key for i in node.parents], ['process-2'])

            elif id_ == "process-2":
                self.assertEqual(node.key, "process-2")
                self.assertEqual(node.attrib['name'], "流程节点2")
                self.assertEqual(len(node.childrens), 2)
                self.assertEqual(len(node.parents), 1)

                self.assertIn("process-1", [i.key for i in node.childrens])
                self.assertIn("process-6", [i.key for i in node.childrens])

                self.assertEqual([i.key for i in node.parents], ["process-3"])

            elif id_ == "process-3":
                self.assertIsInstance(node, DecisionNode)
                self.assertEqual(node.key, "process-3")
                self.assertEqual(node.attrib['name'], '流程节点3')
                self.assertEqual(len(node.childrens), 1)
                self.assertEqual(len(node.parents), 2)

                self.assertEqual([i.key for i in node.childrens], ["process-2"])
                self.assertIn("process-4", [i.key for i in node.parents])
                self.assertIn("process-5", [i.key for i in node.parents])
            elif id_ == "process-4":
                self.assertIsInstance(node, ProcessNode)
                self.assertEqual(node.key, "process-4")
                self.assertEqual(node.attrib['name'], '流程节点4')
                self.assertEqual(len(node.childrens), 1)
                self.assertEqual(len(node.parents), 0)

                self.assertEqual([i.key for i in node.childrens], ["process-3"])
                self.assertEqual([i.key for i in node.parents], [])
            elif id_ == "process-5":
                self.assertIsInstance(node, ProcessNode)
                self.assertEqual(node.key, "process-5")
                self.assertEqual(node.attrib['name'], '流程节点5')
                self.assertEqual(len(node.childrens), 1)
                self.assertEqual(len(node.parents), 1)

                self.assertEqual([i.key for i in node.childrens], ["process-3"])
                self.assertEqual([i.key for i in node.parents], ["process-6"])
            elif id_ == "process-6":
                self.assertIsInstance(node, DecisionNode)
                self.assertEqual(node.key, "process-6")
                self.assertEqual(node.attrib['name'], '流程节点6')
                self.assertEqual(len(node.childrens), 1)
                self.assertEqual(len(node.parents), 1)

                self.assertEqual([i.key for i in node.childrens], ["process-5"])
                self.assertEqual([i.key for i in node.parents], ["process-2"])

    def test_current_available_transition(self):
        work_engine = WorkflowEngine("./sample2.xml")
        for node_id in ("process-1", "process-2", "process-3", "process-4", "process-5", "process-6"):
            result = work_engine.current_available_transition(node_id=node_id)
            print(node_id.center(60, '-'))
            pprint(result, indent=2)

    def test_next_step(self):
        work_engine = WorkflowEngine("./sample2.xml")
        for node_id in ("process-1", "process-2", "process-3", "process-4", "process-5", "process-6"):
            if node_id == "process-1":
                self.assertIsInstance(
                    work_engine.next_step(node_id=node_id), ProcessNode
                )
                self.assertEqual(
                    work_engine.next_step(node_id=node_id).key, "process-2"
                )
            elif node_id == "process-2":
                self.assertIsInstance(
                    work_engine.next_step(node_id=node_id), DecisionNode
                )
                self.assertEqual(
                    work_engine.next_step(node_id=node_id).key,
                    "process-3"
                )
            elif node_id == "process-3":
                # 获取下一个节点，不是DecisionNode
                self.assertIsInstance(
                    work_engine.next_step(node_id=node_id, condition=True), ProcessNode
                )
                self.assertIsInstance(
                    work_engine.next_step(node_id=node_id, condition=False), ProcessNode
                )
                self.assertIsNone(
                    work_engine.next_step(node_id=node_id, condition=None)
                )
                self.assertIsNone(
                    work_engine.next_step(node_id=node_id, condition=1)  # 类型不相同
                )

                # 条件为True的时候
                self.assertEqual(
                    work_engine.next_step(node_id=node_id, condition=True).key,
                    "process-4"
                )
                # 条件为False的时候
                self.assertEqual(
                    work_engine.next_step(node_id=node_id, condition=False).key,
                    "process-5"
                )
            elif node_id == "process-4":
                self.assertIsNone(
                    work_engine.next_step(node_id=node_id)
                )
            elif node_id == "process-5":
                self.assertIsInstance(
                    work_engine.next_step(node_id=node_id), DecisionNode
                )
                self.assertEqual(
                    work_engine.next_step(node_id=node_id).key,
                    "process-6"
                )
            elif node_id == "process-6":
                # 获取下一个节点，不是DecisionNode
                self.assertIsInstance(
                    work_engine.next_step(node_id=node_id, condition=False), ProcessNode
                )
                self.assertIsNone(
                    work_engine.next_step(node_id=node_id, condition=None)
                )
                self.assertIsNone(
                    work_engine.next_step(node_id=node_id, condition=1)  # 类型不相同
                )

                # 条件为True的时候，因为没有下一个node
                self.assertIsNone(
                    work_engine.next_step(node_id=node_id, condition=True),
                )
                # 条件为False的时候
                self.assertEqual(
                    work_engine.next_step(node_id=node_id, condition=False).key,
                    "process-2"
                )
