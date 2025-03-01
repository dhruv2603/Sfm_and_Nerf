import numpy as np
import os
import cv2
import random
from LinearPnp import LinearPnP


def PnPRANSAC(X_i, x_i, K, N=5000, Tau=10):
    """
    Compute RANSAC for the new image by calculating the reprojection error.
    Get the inlier points with repect to the world coordinates.
    Inputs: X_i - (N,3) ndarray of World points
            x_i - (N,2) ndarray of image pixels
            K   - Intrinsic camera matrix
            Tau - Error threshold
    """
    F_identity = np.eye(3)
    Identity = np.hstack([np.eye(3), np.zeros((3, 1))])  # 3x4 matri
    aux_last_element_homogeneous = np.array([[0.0, 0.0, 0.0, 1.0]])

    # Current length of inlier list
    n = 0
    S_in = []
    # Run for 1000 iterations
    row_indices = list(range(X_i.shape[0]))
    X_i = np.hstack((X_i, np.ones((X_i.shape[0], 1))))

    x_i_homo = np.hstack((x_i, np.ones((x_i.shape[0], 1))))
    img_x_i = np.linalg.inv(K) @ x_i_homo.T
    # norm_x_i = (img_x_i / img_x_i[2, :]).T
    norm_x_i = (img_x_i).T

    for i in range(N):
        # Choose 6 correspondences of X-i and x-i randomly
        random_row_indices = random.sample(row_indices, 6)
        X_i_sample = X_i[random_row_indices][:, :3]
        x_i_sample = x_i[random_row_indices]
        R, C = LinearPnP(X_i_sample, x_i_sample, K)
        # print(R)
        T_2 = np.vstack(
            (
                np.hstack((R, C.reshape((3, 1)))),  # shape: (3,4)
                aux_last_element_homogeneous,
            )
        )
        # Camera 2 projection matrix: [R | t]
        P = K @ F_identity @ Identity @ T_2
        # P = R.T @ np.hstack((np.eye(3), -C.reshape(-1, 1)))
        # print(X_i_sample)
        S = []
        for j in range(X_i.shape[0]):
            # e = (x_i[j][0] - (P[:,0].T@X_i[j,:])/(P[:,2].T@X_i[j,:]))**2 + (x_i[j][1] - (P[:,1].T@X_i[j,:])/(P[:,2].T@X_i[j,:]))**2
            X_proj_u = np.dot(P[0], X_i[j, :].T)
            X_proj_v = np.dot(P[1], X_i[j, :].T)
            denom = np.dot(P[2], X_i[j, :].T)

            X_proj_norm_u = X_proj_u / denom
            X_proj_norm_v = X_proj_v / denom
            e = (x_i[j][0] - X_proj_norm_u) ** 2 + (x_i[j][1] - X_proj_norm_v) ** 2
            e = np.sqrt(e)
            # print(e)
            if e < Tau:
                S.append(j)
        if len(S) > n:
            S_in = S
            n = len(S)
    return P, S_in, R, C
