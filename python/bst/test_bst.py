# coding=utf-8

from unittest import TestCase
from .bst import BST


class TestBST(TestCase):
    def test_add(self):
        bst = BST()
        for i in (1, 3, 2, 4, 5):
            bst.add(i)

    def test_contain(self):
        bst = BST()
        for i in (1, 3, 2, 4, 5):
            bst.add(i)
        self.assertTrue(bst.contains(3))
        self.assertFalse(bst.contains(6))

    def test_preorder(self):
        bst = BST()
        for i in (1, 3, 2, 4, 5):
            bst.add(i)
        bst.pre_order()

    def test_inorder(self):
        bst = BST()
        for i in (1, 3, 2, 4, 5):
            bst.add(i)
        bst.in_order()

    def test_postorder(self):
        bst = BST()
        for i in (1, 3, 2, 4, 5):
            bst.add(i)
        bst.post_order()

    def test_level_order(self):
        bst = BST()
        for i in (5, 3, 6, 2, 4, 8):
            bst.add(i)
        bst.level_order()

    def test_del_min(self):
        bst = BST()
        for i in (5, 3, 6, 2, 4, 8):
            bst.add(i)

        e = bst.del_min()
        print(bst.level_order())

    def test_del_max(self):
        bst = BST()
        for i in (5, 3, 6, 2, 4, 8):
            bst.add(i)

        bst.del_max()
        print(bst.level_order())
