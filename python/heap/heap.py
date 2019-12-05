# coding: utf-8
from typing import List


class Heap(object):
    def __init__(self, capacity: int):
        """
        :param capacity: 多大容量的一个数组
        """
        print(f"capacity:{capacity}")
        self.data: List[int] = []

    @property
    def size(self) -> int:
        return len(self.data)

    def is_empty(self) -> bool:
        return self.size == 0

    @staticmethod
    def _parent(index: int) -> int:
        """
        返回index对应的parent index

        由于索引为0的也使用，这里的计算公式发生了改变，同理试用于获取left,right child.
        :param index:
        :return:
        """
        if index == 0:
            raise ValueError("index 0 does't have parent")
        return (index - 1) // 2

    @staticmethod
    def _left_child(index: int) -> int:
        """
        返回index对应的left child index.
        :param index:
        :return:
        """
        return index * 2 + 1

    @staticmethod
    def _right_child(index: int) -> int:
        """
        返回index对应的right child index.
        :param index:
        :return:
        """
        return (index + 1) * 2

    def add(self, e: int):
        """
        添加元素
        :param e:
        :return:
        """
        self.data.append(e)
        self.shift_up(len(self.data) - 1)

    def shift_up(self, index: int):
        while index > 0 and self.data[self._parent(index)] < self.data[index]:
            # swap index and index parent data.
            self.data[index], self.data[self._parent(index)] = self.data[self._parent(index)], self.data[index]
            index = self._parent(index)

    def extract_max(self):
        """
        获取最大值
        :return:
        """
        try:
            e = self.data[0]
        except IndexError:
            raise IndexError('empty data.')

        # 将最后一个元素放到第一个元素位置，然后删除最后一个元素
        self.data[0] = self.data[len(self.data) - 1]
        self.data.pop()
        # 重新计算位置
        self.shift_down(0)
        return e

    def shift_down(self, index: int):
        while self._left_child(index) < len(self.data):
            j = self._left_child(index)
            if j + 1 < len(self.data) and self.data[j + 1] > self.data[j]:
                j = self._right_child(index)

            if self.data[index] >= self.data[j]:
                break
            self.data[index], self.data[j] = self.data[j], self.data[index]
            index = j
