# -*- coding: utf8 -*-
#

# -*- coding:utf-8 -*-
import copy


class ListNode:
    __slots__ = ("val", "next")

    def __init__(self, x):
        self.val = x
        self.next = None

    def __repr__(self):
        return f'Node:{self.val}'


class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # init
        root = ListNode(x=pHead[0])
        cur = root
        for v in pHead[1:]:
            tmp = ListNode(x=v)
            cur.next = tmp
            cur = tmp


        cur = root
        pre = None
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre

a = [1, 2, 3]
b = Solution().ReverseList(a)
print(b)
