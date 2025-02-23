import numpy as np
import scipy
def EssentialMatrixFromFundamentalMatrix(K, F):
    E = np.matmul(np.matmul(np.transpose(K),F),K)
    U,_,V = scipy.linalg.svd(E)
    D = np.array([[1,0,0],
                    [0,1,0],
                    [0,0,0]])
    E = np.matmul(U,np.matmul(D,V))
    return E