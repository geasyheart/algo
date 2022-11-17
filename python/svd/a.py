# -*- coding: utf8 -*-
import numpy as np

# ################## 使用svd进行分解
A = np.random.randn(3, 2)
U, sigma, Vt = np.linalg.svd(A)
print(U.shape, sigma.shape, Vt.shape)

# ################# 重建矩阵
# s 向量必须使用 diag() 函数转换成对角矩阵。默认情况下，这个函数将创建一个相对于原来矩阵的 m×m 的方形矩阵。这是有问题的，因为该矩阵的尺寸并不符合矩阵乘法的规则，即一个矩阵的列数必须等于后一个矩阵的行数。
sigma2 = np.zeros(A.shape)
sigma2[:A.shape[1], :A.shape[1]] = np.diag(sigma)

B = U.dot(sigma2).dot(Vt)

# 设置精度，否则不相等，可以将A.tolist()、B.tolist()后看到
print(np.round(A, 8) == np.round(B, 8))
