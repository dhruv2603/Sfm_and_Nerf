import numpy as np
import scipy
import scipy.linalg

def LinearTriangulation(dl, K, R, C):
    P1 = np.matmul(K,np.array([[1,0,0,0],
                               [0,1,0,0],
                               [0,0,1,0]]))
    P2 = np.matmul(K,np.hstack((R,C)))
    X_list = []
    count  = 0
    for feature in dl:
        x1 = np.array([float(feature[0]), float(feature[1]),1])
        x2 = np.array([float(feature[2]), float(feature[3]), 1])
        x1_skew = np.array([[0, -x1[2], x1[1]],
                            [x1[2], 0, -x1[0]],
                            [-x1[1], x1[0], 0]])
        x2_skew = np.array([[0, -x2[2], x2[1]],
                            [x2[2], 0, -x2[0]],
                            [-x2[1], x2[0], 0]])
        a = np.matmul(x1_skew,P1)
        b = np.matmul(x2_skew,P2)
        A = np.vstack((a,b))
        _,_,V = scipy.linalg.svd(A)
        X_H = V[-1]
        X = np.array([X_H[0], X_H[1], X_H[2]])
        X_list.append(X)
        val = np.matmul(np.transpose(R[:,2]),X-C)
        if val > 0:
            count += 1
    return X_list, count