# coding=utf-8

from unittest import TestCase
from .linkedlist import LinkedList


class TestLinkedList(TestCase):
    def test_linked_list_1(self):
        linked_list = LinkedList()
        linked_list.push(1)
        linked_list.push(2)
        linked_list.push(3)

        print(str(linked_list))
