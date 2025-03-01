import numpy as np
import scipy
import scipy.linalg


def LinearTriangulation(K,C1,R1,C2,R2,uv_1, uv_2):
    I = np.identity(3)
    C1 = np.array(C1).reshape((3,1))
    C2 = np.array(C2).reshape((3,1))
    
    # Projection Matrix for Camera 1
    P1 = K @ R1 @ np.hstack((I, -C1))
    # Projection Matrix for Camera 2
    P2 = K @ R2 @ np.hstack((I, -C2))
    
    p1T = P1[0, :].reshape((1,4))
    p2T = P1[1, :].reshape((1,4))
    p3T = P1[2, :].reshape((1,4))
    
    p1T_hat = P2[0, :].reshape((1,4))
    p2T_hat = P2[1, :].reshape((1,4))
    p3T_hat = P2[2, :].reshape((1,4))

    X_list = []

    for i in range(uv_1.shape[0]):
        x = uv_1[i,0]
        y = uv_2[i,1]

        x_hat = uv_2[i,0]
        y_hat = uv_2[i,1]

        A = np.vstack((y*p3T - p2T, p1T - x*p3T, y_hat*p3T_hat - p2T_hat, p1T_hat - x_hat*p3T_hat))
        
        A = A.reshape((4,4))
        U,_, V = np.linalg.svd(A)
        V = V.T
        X = V[:,-1]
        X = X/X[-1]
        X = X[:3]
        X_list.append(X)

    return X_list


def triangulatePoints(x1_h, x2_h, P1, P2):
    """
    Triangulates points using linear triangulation.

    Parameters:
        x1_h : (3, N) numpy array of homogeneous image coordinates from camera 1.
        x2_h : (3, N) numpy array of homogeneous image coordinates from camera 2.
        P1   : (3, 4) projection matrix for camera 1.
        P2   : (3, 4) projection matrix for camera 2.

    Returns:
        X_4xN : (4, N) numpy array of homogeneous 3D coordinates.
    """
    N = x1_h.shape[1]
    X_4xN = np.zeros((4, N))

    for i in range(N):
        # Construct the A matrix for the i-th point.
        A = np.array(
            [
                x1_h[1, i] * P1[2, :] - P1[1, :],
                P1[0, :] - x1_h[0, i] * P1[2, :],
                x2_h[1, i] * P2[2, :] - P2[1, :],
                P2[0, :] - x2_h[0, i] * P2[2, :],
            ]
        )

        # Solve A * X = 0 using SVD.
        _, _, Vh = np.linalg.svd(A)
        X = Vh[-1, :]  # Last row of Vh corresponds to the smallest singular value.

        # Store the homogeneous 3D point.
        X_4xN[:, i] = X

    return X_4xN
