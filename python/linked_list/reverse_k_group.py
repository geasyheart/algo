# -*- coding: utf8 -*-
#
from typing import Optional


class ListNode(object):
    def __init__(self, val: int, next: Optional['ListNode']):
        self.val = val
        self.next = next

    def __repr__(self):
        return f'ListNode[{self.val}]'


class Solution(object):
    def reverse_k_group(self, head: ListNode, k: int) -> ListNode:
        prev = None
        curr = head
        times = k

        while times > 0:
            if curr is None:
                return self.restore(prev)
            next: ListNode = curr.next
            curr.next = prev
            prev = curr
            curr = next

            times -= 1

        head.next = self.reverse_k_group(curr, k)
        return prev

    def restore(self, head: ListNode) -> ListNode:
        prev = None
        curr = head

        while curr is not None:
            next: ListNode = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev


if __name__ == '__main__':
    a, k = [1, 2, 3, 4, 5], 2
    a_nodes = [ListNode(val=_a, next=None) for _a in a]
    for i, node in enumerate(a_nodes):
        try:
            next_node = a_nodes[i+1]
            node.next = next_node
        except IndexError:
            pass
    result = Solution().reverse_k_group(a_nodes[0], k=2)

    cur = result
    while cur is not None:
        print(cur.val)
        cur = cur.next

