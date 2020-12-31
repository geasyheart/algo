# -*- coding: utf8 -*-

# https://leetcode-cn.com/problems/triangle/solution/kong-jian-fu-za-du-wei-onde-wei-te-bi-suan-fa-jie-/

"""
[
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]
"""

import numpy as np


class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """

        if len(triangle) == 0:
            return 0

        viterbi = [-1] * len(triangle)

        for index, elements in enumerate(triangle):
            # print('elements--', elements)
            if index == 0:  # 如果是第一行,直接给第一个值赋值
                viterbi[0] = elements[0]
            else:  # 如果不是第一行，则根据当前的局部最短路径分别寻找到达当前位置的最短路径

                tmp_viterbi = np.copy(viterbi)
                # print('tmp_viterbi---', tmp_viterbi)
                for i, element in enumerate(elements):
                    if i == 0:  # 每一行的0元素位置只能从上一行的0元素位置处过来
                        viterbi[i] = tmp_viterbi[i] + element
                        print('tmp_viterbi[{}]={}, element={}'.format(i, tmp_viterbi[i], element))
                        continue

                    if i == len(elements) - 1:  # 当前行的最后一个元素只能从上一行的最后一个位置处到达
                        viterbi[i] = tmp_viterbi[i - 1] + element
                        # print('tmp_viterbi---', tmp_viterbi)
                        continue

                    # 0 < i < len(element) - 1
                    viterbi[i] = min(tmp_viterbi[i - 1], tmp_viterbi[i]) + element
                    # print('tmp_viterbi---', tmp_viterbi)

            # print(viterbi)

            # 如果已经到了最后一行，则找出其中最小的值返回即可
            if index == len(triangle) - 1:
                return min(viterbi)


if __name__ == '__main__':
    Solution().minimumTotal([
        [2],
        [3, 4],
        [6, 5, 7],
        [4, 1, 8, 3]
    ])
