import random
import time


class BenchMarkUnionFind(object):
    def test_uf(self, uf, m: int):
        size: int = uf.get_size()
        start_time: float = time.time()

        for i in range(m):
            a = random.randint(1, size)
            b = random.randint(1, size)

            uf.union_elements(a, b)

        for i in range(m):
            a = random.randint(1, size)
            b = random.randint(1, size)
            uf.is_connected(a,b)



        end_time:float   = time.time()
        return end_time - start_time
