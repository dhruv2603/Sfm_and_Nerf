import numpy as np
import scipy
import os
from NonlinearTriangulation import (
    init_optimization_variables,
    cameraCalibrationCasADi,
    init_optimization_pose,
    cameraCalibrationPose,
)
import scipy.linalg
from LinearTriangulation import triangulatePoints, LinearTriangulation


def LinearPnP(X_i, x_i, K):
    """
    Calculate the P matrix
    Inputs: X_i - (6,3) ndarray of World points
            x_i - (6,2) ndarray of image pixels
            K   - Intrinsic matrix
    Outputs:R  - Rotation Matrix
            C  - Camera center wrt to world frame
    """
    x_i_homo = np.hstack((x_i, np.ones((x_i.shape[0], 1))))
    img_x_i = np.linalg.inv(K) @ x_i_homo.T
    norm_x_i = (img_x_i / img_x_i[2, :]).T
    # norm_x_i = img_x_i.T
    A = np.empty([0, 12])
    I = np.eye(3)
    for i, data in enumerate(x_i_homo):
        # for i, data in enumerate(norm_x_i):
        u = float(data[0])
        v = float(data[1])
        x = X_i[i][0]
        y = X_i[i][1]
        z = X_i[i][2]
        row = np.array(
            [
                [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u],
                [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v],
            ]
        )
        A = np.vstack((A, row))
    _, _, V = scipy.linalg.svd(A)
    P = V[-1]
    P = np.reshape(P, (3, 4))
    R_init = np.linalg.inv(K) @ P[:, :3]
    # R_init = P[:, :3]
    Ur, Dr, Vr = scipy.linalg.svd(R_init)
    R = np.matmul(Ur, Vr)
    gamma = Dr[0]
    # T = P[:,3]
    if np.linalg.det(R) < 0:
        R = -R
        # T = -T
        C = -np.linalg.inv(K) @ P[:, 3] / gamma
        # C = P[:, 3] / gamma
    else:
        R = R
        # T = -T
        C = np.linalg.inv(K) @ P[:, 3] / gamma
        # C = P[:, 3] / gamma

    return R, C


def TriangulationPnp(X_i, x_j, x_i, inlier_idxs, K, translation_init, rotation_init):

    # Linear Pnp
    world_points_data = np.vstack(
        (X_i[inlier_idxs, :].T, np.ones((1, X_i[inlier_idxs, :].shape[0])))
    )
    ## initial Condition
    x_init = init_optimization_pose(translation_init, rotation_init)
    # Optimization problem
    t_new, R_new = cameraCalibrationPose(
        x_i[inlier_idxs, :].T, K, x_init, world_points_data[0:3, :]
    )
    P1 = K @ np.hstack((rotation_init, translation_init.reshape(3, 1)))
    P2 = K @ np.hstack((R_new, t_new.reshape(3, 1)))

    X = triangulatePoints(x_j[inlier_idxs, :].T, x_i[inlier_idxs, :].T, P1, P2)
    X = X / X[3, :]

    return X, t_new, R_new
