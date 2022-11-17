# -*- coding: utf8 -*-
#
import numpy as np
from scipy.sparse.linalg import svds
from skimage import io


# 对降维的应用

def plotImg(imgMat):
    import matplotlib.pyplot as plt

    plt.imshow(imgMat, cmap=plt.cm.gray)
    plt.show()


def getImgAsMatFromFile(filename):
    img = io.imread(filename, as_gray=True)
    return np.mat(img)


def recover(img, k):
    uu, ss, vtt = svds(img, k=k)
    return np.dot(np.dot(uu, np.diag(ss)), vtt)


def main():
    """
    https://zhuanlan.zhihu.com/p/36546367
    可以看到，k的不同，生成的图片清晰度也不同
    """
    A = getImgAsMatFromFile('img1.webp')
    # plotImg(A)

    A_2 = recover(A, k=30)
    io.imsave('img2.png', A_2)

    A_3 = recover(A, k=10)
    io.imsave('img3.png', A_3)


if __name__ == '__main__':
    main()
