# -*- coding: utf8 -*-
#
# SVD降维
# PCA也可以使用SVD进行降维
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD


def a():
    """
    降维
    https://www.jiqizhixin.com/articles/0301
    """
    A = np.random.randn(3, 10)
    U, s, V = np.linalg.svd(A)
    Sigma = np.zeros(A.shape)
    Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)
    # 降维
    k = 2
    Sigma = Sigma[:, :k]
    V = V[:k, :]
    B = U.dot(Sigma.dot(V))
    T = U.dot(Sigma)
    T = A.dot(V.T)

    # 比较和TruncatedSVD的实现
    svd = TruncatedSVD(n_components=k)
    svd.fit(A)
    result = svd.transform(A)

    print(T)
    print(result)
    # 可以看到，结果得到的值与上面人工计算的结果一致，但某些值的符号不一样。
    # 由于所涉及的计算的性质以及所用的基础库和方法的差异，可以预见在符号方面会存在一些不稳定性。
    # 只要对该变换进行训练以便复用，这种不稳定性在实践中应该不成问题。


def b():
    """
    降低稀疏性
    """
    A = np.random.randn(3, 10)
    U, s, V = np.linalg.svd(A)
    Sigma = np.zeros(A.shape)
    Sigma[:A.shape[0], :A.shape[0]] = np.diag(s)

    k = 2
    Sigma = Sigma[:, :k]
    V = V[:k, :]
    B = U.dot(Sigma.dot(V))
    print(B)

    # ################### 通过scipy实现
    # uu, ss, vtt = svds(csr_matrix(A), k=k)
    uu, ss, vtt = svds(A, k=k)

    B_new = np.dot(np.dot(uu, np.diag(ss)), vtt)
    print(np.round(B, 8) == np.round(B_new, 8))

    # 可以看到，重构的矩阵B_new和B（自己实现的降维方式）结果一样
    # A和B_new的区别：B_new稀疏的稀疏性没有A强，至于k的选择，就是一个超参了，可以看c.py


if __name__ == '__main__':
    b()
