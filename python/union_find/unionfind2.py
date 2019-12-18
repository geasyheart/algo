class UnionFind2(object):
    def __init__(self, capacity: int):
        self.parent = [i for i in range(capacity)]

    def get_size(self):
        return len(self.parent)

    def _find(self, p: int) -> int:
        if p < 0 or p >= len(self.parent):
            raise ValueError("error")
        while p != self.parent[p]:
            p = self.parent[p]
        return p

    def is_connected(self, p: int, q:int) -> bool:
        return self._find(p) == self._find(q)

    def union_elements(self, p:int, q:int):
        p_root: int = self._find(p)
        q_root: int = self._find(q)

        if p_root == q_root:
            return
        # 指向q_root节点
        self.parent[p_root] = q_root
