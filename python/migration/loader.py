# coding=utf-8
from typing import Dict, List, Union

from migration.graph import MigrationGraph, Node

from orm import Table


class MigrationLoader(object):
    def __init__(self, load=True):
        self.local_migrations: Dict[int, Table] = {}
        self.applied_migrations: Dict[int, Table] = {}
        self.local_graph: Union[MigrationGraph, None] = None
        self.remote_graph: Union[MigrationGraph, None] = None
        if load:
            self.build_local_graph()
            self.build_remote_graph()

    def build_local_graph(self):
        self.load_local_record()

        self.local_graph = MigrationGraph()

        for key, table in self.local_migrations.items():
            self.local_graph.add_node(key, table)

        for key, table in self.local_migrations.items():
            self.local_graph.add_dependencies(key, table)

        self.local_graph.validate_consistency()
        self.local_graph.ensure_not_cyclic()

    def build_remote_graph(self):
        self.load_remote_record()
        self.remote_graph = MigrationGraph()

        for key, table in self.applied_migrations.items():
            self.remote_graph.add_node(key, table)

        for key, table in self.applied_migrations.items():
            self.remote_graph.add_dependencies(key, table)

        self.remote_graph.validate_consistency()
        self.remote_graph.ensure_not_cyclic()

    def make_plan(self):
        if len(self.local_graph.node_map) != len(self.remote_graph.node_map):
            raise ValueError("nodes count not equal same.")
        # 层级遍历，获取diff
        local_level_node: List[List[Node]] = self.local_graph.iterative_level_order()
        changes = []
        for level, targets in enumerate(local_level_node[1:]):
            for target in targets:
                if target.parent.key != self.applied_migrations[target.key].parent:
                    changes.append(target)
        return changes

    def load_local_record(self):
        """
        {
        id: parent
        }
        :return:
        """
        self.local_migrations = {
            1: Table(pk=1, name="root", parent=1),
            5: Table(pk=123, name="node5", parent=1),
            4: Table(pk=456, name="node4", parent=5),
            2: Table(pk=234, name="node2", parent=1),
            3: Table(pk=345, name="node3", parent=1),
        }

    def load_remote_record(self):
        self.applied_migrations = {
            1: Table(pk=1, name="root", parent=1),
            5: Table(pk=123, name="node5", parent=1),
            4: Table(pk=456, name="node4", parent=1),
            2: Table(pk=234, name="node2", parent=1),
            3: Table(pk=345, name="node3", parent=1),
        }

