# coding=utf-8


class Node(object):
    def __init__(self, e, next_node):
        self.e = e
        self.next = next_node

    def __repr__(self):
        return f"<Node> {self.e}"


class LinkedList(object):
    def __init__(self):
        self.head = None
        self.size = 0

    def push(self, e):
        self.head = Node(e, self.head)
        self.size += 1

    def reversed(self):
        prev = None
        cur = self.head
        while cur is not None:
            next_node = cur.next
            cur.next = prev
            prev = cur
            cur = next_node
        self.head = prev


    def __repr__(self):
        rs = []
        cur = self.head
        while cur is not None:
            rs.append(str(cur.e))
            cur = cur.next
        return f"<LinkedList> {'->'.join(rs)}"
