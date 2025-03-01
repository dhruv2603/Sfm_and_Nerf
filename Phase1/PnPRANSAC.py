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
    X_i_homo = np.hstack((X_i, np.ones((X_i.shape[0], 1))))
    x_i_homo = np.hstack((x_i, np.ones((x_i.shape[0], 1))))

    # Current length of inlier list
    n = 0
    S_in = []

    for i in range(N):
        # Choose 6 correspondences of X-i and x-i randomly
        random_row_indices = np.random.choice(X_i_homo.shape[0], 6, replace=False)
        X_i_sample = X_i_homo[random_row_indices]
        x_i_sample = x_i[random_row_indices]

        R, C = LinearPnP(X_i_sample, x_i_sample, K)

        P = K @ np.hstack((R,C.reshape(-1,1)))

        X_proj = (P @ X_i_homo.T).T
        X_proj = X_proj/X_proj[:,-1].reshape(-1,1)
        
        e = np.sum((X_proj[:,:-1] - x_i_homo[:,:-1])**2, axis=1)
        inlier_idxs = e < Tau
        inliers = np.sum(inlier_idxs)
        if inliers > n:
            n = inliers
            R_best = R
            C_best = C
            S_in = inlier_idxs

    return S_in, R, C
