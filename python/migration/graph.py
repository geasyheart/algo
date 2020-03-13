# coding=utf-8

from functools import total_ordering
from typing import Dict

from migration.orm import Table


class NodeNotFoundError(LookupError):
    """An attempt on a node is made that is not available in the graph."""

    def __init__(self, message, node, origin=None):
        self.message = message
        self.origin = origin
        self.node = node

    def __str__(self):
        return self.message

    def __repr__(self):
        return "NodeNotFoundError(%r)" % (self.node,)


class CircularDependencyError(Exception):
    """There's an impossible-to-resolve circular dependency."""
    pass


@total_ordering
class Node(object):
    def __init__(self, key):
        self.key = key
        self.children = set()
        self.parent = None

    def __eq__(self, other):
        return self.key == other

    def __lt__(self, other):
        return self.key < other

    def __hash__(self):
        return hash(self.key)

    def __getitem__(self, item):
        return self.key[item]

    def __repr__(self):
        return f"<Node: {self.key}>"

    def add_child(self, child):
        self.children.add(child)

    def add_parent(self, parent):
        self.parent = parent

class DummyNode(Node):
    def __init__(self, key, origin, error_message):
        super().__init__(key)
        self.origin = origin
        self.error_message = error_message

    def raise_error(self):
        raise NodeNotFoundError(self.error_message, self.key, origin=self.origin)


class MigrationGraph(object):
    def __init__(self):
        self.node_map: Dict[int, Node] = {}
        self.node_table: Dict[int, Table] = {}

    def add_node(self, key: int, table: Table):
        assert key not in self.node_map
        node = Node(key)
        self.node_map[key] = node
        self.node_table[key] = table

    def add_dummy_node(self, key: int, table: Table, error_message):
        node = DummyNode(key=key, origin=table, error_message=error_message)
        self.node_map[key] = node
        self.node_table[key] = table

    def add_dependencies(self, key: int, table: Table):
        # 忽略掉自引用
        if key == table.parent:
            return
        # 引入虚拟节点
        if key not in self.node_table:
            self.add_dummy_node(key, table, f"dummy node: {key}")
        if table.parent not in self.node_table:
            self.add_dummy_node(table.parent, table, f"dummy node: {key}")
        self.node_map[key].add_parent(self.node_map[table.parent])
        self.node_map[table.parent].add_child(self.node_map[key])

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
                for child in self.node_map[top].children:
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

    def iterative_dfs(self, start, forwards=True):
        """Iterative depth-first search for finding dependencies."""
        visited = []
        visited_set = set()
        stack = [(start, False)]
        while stack:
            # forward True时从根结点往下找
            # forward False时从叶子节点往上找
            node, processed = stack.pop()
            if node in visited_set:
                pass
            elif processed:
                visited_set.add(node)
                visited.append(node.key)
            else:
                stack.append((node, True))
                stack += [(n, False) for n in sorted(node.parents if forwards else node.children)]
        return visited

    ROOT = 1

    def iterative_level_order(self):
        visited = [
            [self.node_map[self.ROOT]],
        ]
        q = [self.ROOT]
        while q:
            node = q.pop()
            children = self.node_map[node].children
            if not children:
                continue
            visited.append(children)
            q.extend(children)
        return visited

    def __repr__(self):
        nodes, edges = self._nodes_and_edges()
        return '<%s: nodes=%s, edges=%s>' % (self.__class__.__name__, nodes, edges)

    def _nodes_and_edges(self):
        return len(self.node_table), sum(len(node.parents) for node in self.node_map.values())


