import numpy as np
import scipy
import os

import scipy.linalg

def LinearPnP(X,dl,K):
    A = np.empty([0,12])
    for i,data in enumerate(dl):
        u = float(data[0])
        v = float(data[1])
        x = X[i][0]
        y = X[i][1]
        z = X[i][2]
        row = np.array([[x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u],
                        [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v]])
        np.vstack((A,row))
    
    _,_,V = scipy.linalg.svd(A)
    P = V[-1]
    P = np.reshape(P,(3,4))
    R_init = np.matmul(np.linalg.pinv(K),P[:,:3])
    Ur,Dr,Vr = scipy.linalg.svd(R_init)
    R = np.matmul(Ur,Vr)
    gamma = Dr[0]
    T = np.matmul(np.linalg.pinv(K),P[:,3])/gamma

    return R,T

