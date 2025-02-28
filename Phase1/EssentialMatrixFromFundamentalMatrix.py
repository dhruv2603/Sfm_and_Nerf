import numpy as np
import scipy


def EssentialMatrixFromFundamentalMatrix(K, F):
    E = np.matmul(np.matmul(np.transpose(K), F), K)
    return E
