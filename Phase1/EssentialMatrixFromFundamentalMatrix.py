import numpy as np
import scipy


def EssentialMatrixFromFundamentalMatrix(K, F):
    """
    Compute Essential Matrix
    Inputs: K - intrinsic calibration matrix
            F - Fundamental matrix
    Output: E - Essential Matrix
    """

    E = K.T @ F @ K
    U,_,V = np.linalg.svd(E)
    E = U @ np.diag([1, 1, 0]) @ V
    return E
