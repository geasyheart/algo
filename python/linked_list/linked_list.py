# coding=utf-8


class Node(object):
    def __init__(self, e, next_node):
        self.e = e
        self.next_node: Node = next_node

    def __repr__(self):
        return f"<Node> {self.e}"


class LinkedList(object):
    def __init__(self, ):
        self.head = Node(None, None)
        self.size = 0

    def get_size(self):
        return self.size

    @property
    def empty(self):
        return self.get_size() == 0

    def add(self, index: int, e):
        if index < 0 or index > self.size:
            raise IndexError("error")
        prev = self.head
        for i in range(index):
            prev = prev.next_node
        prev.next_node = Node(e, prev.next_node)
        self.size += 1

    def add_first(self, e):
        self.add(0, e)

    def add_last(self, e):
        self.add(self.size, e)

    def get(self, index):
        cur = self.head.next_node

        for i in range(index):
            cur = cur.next_node
        return cur.e

    def __repr__(self):
        rs = []
        cur = self.head
        for i in range(self.size):
            cur = cur.next_node
            rs.append(cur.e)
        return "->".join([str(i) for i in rs])

    def reverse(self):
        prev = self.head
        current = self.head.next_node
        while current is not None:
            next = current.next_node
            current.next_node = prev
            prev = current
            current = next
        self.head = prev
