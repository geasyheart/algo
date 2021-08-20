# -*- coding: utf8 -*-
#
"""
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/lru-cache
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
"""


class LRUCache(object):
    def __init__(self, cap=3):
        self.keys = []
        self.storage = {}
        self.cap = cap

    def get(self, key):
        value = self.storage.get(key)
        if value is None:
            return -1
        self.keys.remove(key)
        self.keys.insert(0, key)
        return value

    def put(self, key, value):
        if len(self.keys) >= self.cap:
            pop_key = self.keys.pop()
            self.storage.pop(pop_key)
        self.keys.insert(0, key)
        self.storage[key] = value


if __name__ == '__main__':
    lrucache = LRUCache(2)
    lrucache.put(1, 1)
    lrucache.put(2, 2)
    assert lrucache.get(1) == 1
    lrucache.put(3, 3)
    assert lrucache.get(2) == -1
    lrucache.put(4, 4)
    assert lrucache.get(1) == -1
    assert lrucache.get(3) == 3
    assert lrucache.get(4) == 4
