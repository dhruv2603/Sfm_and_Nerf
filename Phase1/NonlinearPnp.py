import numpy as np
import scipy
import os
from NonlinearTriangulation import (
    init_optimization_variables,
    cameraCalibrationCasADi,
    init_optimization_pose,
    cameraCalibrationPose,
)


def NonlinearPnpCasadi(
    X,
    x_j,
    x_i,
    inlier_idxs,
    t_new,
    R_new,
    translation_init,
    rotation_init,
    K,
    gain_1,
    gain_2,
):
    x_init = init_optimization_variables(t_new, R_new, X)
    X_opt, C_opt, R_quaternion_opt, distortion_opt = cameraCalibrationCasADi(
        x_j[inlier_idxs, :].T,
        x_i[inlier_idxs, :].T,
        K,
        x_init,
        rotation_init,
        translation_init,
        R_new,
        t_new,
        X,
        gain_1,
        gain_2,
    )
    # Homogenization
    X_4xN_casadi = np.vstack((X_opt, np.ones((1, X_opt.shape[1]))))

    return X_4xN_casadi, C_opt, R_quaternion_opt
