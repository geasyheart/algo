# -*- coding: utf8 -*-
#

import numpy as np

print('二维'.center(60, '-'))
a = np.array([
    [1, 2, 3, 0, 0],
    [1, 2, 3, 4, 0]
])
mask = np.not_equal(a, 0)
print(a[mask])

print('三维'.center(60, '-'))
# mask
a = np.array([[[1., 1., 0, 0],
               [1., 1., 1., 0, ],
               [1., 0, 0, 0]],

              [[1., 1., 1., 0],
               [1., 1., 1., 0],
               [1., 1., 1., 0.]]])

mask = np.any(a, -1)
print(a[mask])
print(a[np.not_equal(a, 0)])
