import numpy as np
import scipy
import scipy.linalg


def LinearTriangulation(dl, K, R, C):
    P1 = np.matmul(K, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
    X_list = []
    P2 = K @ np.hstack((R, C.reshape(3, 1)))
    count = 0
    for feature in dl:
        x1 = np.array([float(feature[0]), float(feature[1]), 1])
        x2 = np.array([float(feature[2]), float(feature[3]), 1])
        x1_skew = np.array([[0, -x1[2], x1[1]], [x1[2], 0, -x1[0]], [-x1[1], x1[0], 0]])
        x2_skew = np.array([[0, -x2[2], x2[1]], [x2[2], 0, -x2[0]], [-x2[1], x2[0], 0]])
        a = np.matmul(x1_skew, P1)
        b = np.matmul(x2_skew, P2)
        A = np.vstack((a, b))
        _, _, V = scipy.linalg.svd(A)
        X_H = V[-1]
        X_H = X_H / X_H[3]
        X = np.array([X_H[0], X_H[1], X_H[2], X_H[3]])
        x_3d = np.array([X_H[0], X_H[1], X_H[2]])
        X_list.append(X)
        check_aux = np.dot(R[:, 2], x_3d - C)
        val = check_aux
        if val > 0:
            count += 1
    return X_list, count


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
