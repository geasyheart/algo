# coding=utf-8
from typing import Hashable, Any


class LRUCache(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """

        self.cache = {}
        self.keys = []
        self.capacity = capacity

    def visit_key(self, key):
        if key in self.keys:
            self.keys.remove(key)
        self.keys.append(key)

    def elim_key(self):
        key = self.keys[0]
        self.keys = self.keys[1:]
        del self.cache[key]

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if not key in self.cache:
            return -1
        self.visit_key(key)
        return self.cache[key]

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if not key in self.cache:
            if len(self.keys) == self.capacity:
                self.elim_key()
        self.cache[key] = value
        self.visit_key(key)


class LRU2Cache(object):
    def __init__(self, capacity: int):
        self.capacity = capacity

        self.cache = dict()
        self.keys = list()

    def resort(self, key):
        if key in self.keys:
            self.keys.remove(key)
        self.keys.append(key)

    def delete_old(self):
        key = self.keys[0]
        self.keys = self.keys[1:]
        self.cache.pop(key)

    def get(self, key: Hashable):
        try:
            value = self.cache[key]
            self.resort(key=key)
            return value
        except KeyError:
            return -1

    def put(self, key: Hashable, value: Any):
        if key not in self.cache:
            if len(self.cache) == self.capacity:
                self.delete_old()

        self.cache[key] = value
        self.resort(key=key)


if __name__ == '__main__':
    lru = LRUCache(2)
    lru.put(1, 1)
    lru.put(2, 2)
    lru.put(1, 2)
    lru.put(3, 3)
