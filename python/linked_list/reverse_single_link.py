# coding=utf-8


class Node(object):
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next_node = next_node

    def __repr__(self):
        return f"<Node: {self.data}>"


def reverse(root: Node):
    # 遍历1
    # if root:
    #     print(root.data)
    #     return reverse(root.next_node)

    # 反转
    pre = root
    cur = root.next_node
    pre.next_node = None
    while cur:
        temp = cur.next_node
        cur.next_node = pre
        pre = cur
        cur = temp
    return pre


if __name__ == '__main__':
    link = Node(1, Node(2, Node(3, Node(4, Node(5, Node(6, Node(7, Node(8, Node(9)))))))))
    root = reverse(link)
    while root:
        print(root.data)
        root = root.next_node
