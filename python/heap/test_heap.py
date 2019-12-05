from unittest import TestCase

from .heap import Heap


class TestHeap(TestCase):
    def test_heap(self):
        heap = Heap(10000)
        for i in (1, 3, 2, 5):
            heap.add(i)
        print(heap.extract_max())
        print(heap.extract_max())
        print(heap.extract_max())
        print(heap.extract_max())
