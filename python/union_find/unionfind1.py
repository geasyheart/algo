class UnionFindV1(object):
    def __init__(self, capacity: int):
        self.ids = [i for i in range(capacity)]

    def get_size(self):
        return len(self.ids)

    def _find(self, index: int):
        if index < 0 or index > len(self.ids):
            raise ValueError("error")
        return self.ids[index]

    def is_connected(self, p: int, q: int) -> bool:
        return self._find(p) == self._find(q)

    def union_elements(self, p: int, q: int):
        p_id: int = self._find(p)
        q_id: int = self._find(q)
        if p_id == q_id:
            return
        for index, ele in enumerate(self.ids):
            if ele == p_id:
                self.ids[index] = q_id


if __name__ == '__main__':
    union_find = UnionFindV1(3)
    union_find.union_elements(1, 2)
    print(union_find.ids)
    print(union_find.is_connected(1, 2))
