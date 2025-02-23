import numpy as np
import scipy

def ExtractCameraPose(E):
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    U,D,V = scipy.linalg.svd(E)
    C1 = U[:,2]
    R1 = np.matmul(U,np.matmul(W,V))
    C2 = -U[:,2]
    R2 = np.matmul(U,np.matmul(W,V))
    C3 = U[:,2]
    R3 = np.matmul(U,np.matmul(np.transpose(W),V))
    C4 = -U[:,2]
    R4 = np.matmul(U,np.matmul(np.transpose(W),V))
    if np.linalg.det(R1) < 0:
        R1 = -R1
        C1 = -C1
        R2 = -R2
        C2 = -C2
        R3 = -R3
        C3 = -C3
        R4 = -R4
        C4 = -C4

    print(np.linalg.det(R1))
    print(np.linalg.det(R2))
    print(np.linalg.det(R3))
    print(np.linalg.det(R4))

    return [R1, R2, R3, R4],[C1, C2, C3, C4]