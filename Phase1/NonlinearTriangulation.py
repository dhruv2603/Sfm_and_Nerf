import numpy as np
import os
import scipy
import scipy.linalg
import casadi as ca
import time
import math
from scipy.spatial.transform import Rotation as R
import cv2 as cv2


def quat_to_rot(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    R = ca.vertcat(
        ca.horzcat(
            1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)
        ),
        ca.horzcat(
            2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)
        ),
        ca.horzcat(
            2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)
        ),
    )
    return R


def cameraCalibrationCasADi(pts1, pts2, A, x_init, R1, t1, R2, t2, x):
    # Ensure pts1 and pts2 are of type float (double precision)
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts2 = np.asarray(pts2, dtype=np.float64)

    # Get size parameters from pts1: assume pts1 is m x N.
    size_optimization_x, size_optimization_y = (
        x.shape
    )  # e.g., m=4, N = number of points

    # d = (size_optimization_x - 1) * size_optimization_y
    d = 2 + 6 + (size_optimization_x - 1) * size_optimization_y
    a_vector = ca.SX.sym("full_estimation", d, 1)

    # Initialize cost to zero.
    cost = 0

    # Distortion coefficients: first 2 elements.
    distortion = a_vector[0:2]
    # Translation: elements 3:5 (MATLAB indices 3:5 -> Python indices 2:5)
    x_trans = a_vector[2:5]
    # Rotation (minimal representation, e.g., a 3-vector): elements 6:8 (Python indices 5:8)
    x_quaternion = a_vector[5:8]
    # The remaining elements (from index 8 onward) represent the 3D points.
    vector_optimization = a_vector[8:]
    # Reshape vector_optimization into a 3 x N matrix.
    x_vector = ca.reshape(vector_optimization, 3, size_optimization_y)

    # --- Map the 3-vector to a quaternion ---
    # Compute norm squared of x_quaternion.
    norm_sq = ca.dot(x_quaternion, x_quaternion)
    q0 = (1 - norm_sq) / (1 + norm_sq)
    q1 = 2 * x_quaternion[0] / (1 + norm_sq)
    q2 = 2 * x_quaternion[1] / (1 + norm_sq)
    q3 = 2 * x_quaternion[2] / (1 + norm_sq)
    quaternion = ca.vertcat(q0, q1, q2, q3)  # quaternion in form [w, x, y, z]
    R2 = quat_to_rot(quaternion)

    t1 = ca.DM(t1)
    R1 = ca.DM(R1)

    t2 = x_trans
    R2 = R2

    U_real1 = pts1[0:2, :]
    U_real2 = pts2[0:2, :]
    U_improved_final_1 = projection(R1, t1, A, x_vector, pts1)
    U_improved_final_2 = projection(R2, t2, A, x_vector, pts2)

    ### --- Compute reprojection error ---
    ## U_real is a numpy array; convert it to CasADi DM.
    error_1 = U_real1 - U_improved_final_1
    error_2 = U_real2 - U_improved_final_2
    ### Reshape error into a column vector.
    error_reshape_1 = ca.reshape(error_1, (2 * error_1.shape[1], 1))
    error_reshape_2 = ca.reshape(error_2, (2 * error_2.shape[1], 1))
    cost = (
        cost + error_reshape_2.T @ error_reshape_2 + error_reshape_1.T @ error_reshape_1
    )

    ### --- Set up and solve the NLP ---
    nlp = {"x": a_vector, "f": cost}
    opts = {"print_time": 0, "ipopt": {"print_level": 0}}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    sol = solver(x0=x_init)
    a_opt_vec = np.array(sol["x"]).flatten()

    ### --- Extract optimized parameters ---
    distortion_opt = a_opt_vec[0:2]
    vector_optimization_opt = a_opt_vec[8:]
    ## Reshape vector_optimization_opt into a (3 x size_optimization_y) array.
    x_vector_opt = np.reshape(
        vector_optimization_opt, (3, size_optimization_y), order="F"
    )
    ### Extract the optimized rotation and translation parts.
    x_quaternion_opt = a_opt_vec[5:8]
    norm_sq_opt = np.dot(x_quaternion_opt, x_quaternion_opt)
    q0_opt = (1 - norm_sq_opt) / (1 + norm_sq_opt)
    q1_opt = 2 * x_quaternion_opt[0] / (1 + norm_sq_opt)
    q2_opt = 2 * x_quaternion_opt[1] / (1 + norm_sq_opt)
    q3_opt = 2 * x_quaternion_opt[2] / (1 + norm_sq_opt)
    quaternion_opt = np.array([q0_opt, q1_opt, q2_opt, q3_opt])

    x_trans_opt = a_opt_vec[2:5]
    R_quaternion_opt = quat_to_rot(quaternion_opt)

    R_quaternion_opt = np.array(R_quaternion_opt)
    x_trans_opt = np.array(x_trans_opt)
    x_vector_opt = np.array(x_vector_opt)
    distortion_opt = np.array(distortion_opt)

    return x_vector_opt, x_trans_opt, R_quaternion_opt, distortion_opt


def projection(R, t, A, x_vector, pts):
    aux_last_element_homogeneous = ca.DM([[0, 0, 0, 1]])
    F_identity = np.eye(3)
    Identity = np.hstack([np.eye(3), np.zeros((3, 1))])  # 3x4 matrix

    T_estimated = ca.vertcat(ca.horzcat(R, t), aux_last_element_homogeneous)

    ones_row = ca.DM.ones(1, x_vector.shape[1])
    x_vector_hom = ca.vertcat(x_vector, ones_row)  # (4 x N)

    F_identity_ca = ca.DM(F_identity)
    Identity_ca = ca.DM(Identity)

    values_normalized = F_identity_ca @ Identity_ca @ T_estimated @ x_vector_hom

    aux_normalization = ca.vertcat(values_normalized[2, :], values_normalized[2, :])
    values_normalized_aux = values_normalized[0:2, :] / aux_normalization

    x_warp = values_normalized_aux
    ones_row_warp = ca.DM.ones(1, x_warp.shape[1])
    x_warp_aux = ca.vertcat(x_warp, ones_row_warp)  # (3 x N)

    U_improved = ca.DM(A) @ x_warp_aux
    U_normalized_aux = ca.vertcat(U_improved[2, :], U_improved[2, :])
    U_improved_final = U_improved[0:2, :] / U_normalized_aux

    return U_improved_final


def init_optimization_variables(translation, rotation, points):
    # Computethe minimal rotation representation: x_quaternion = (x,y,z)/w.

    rotations = R.from_matrix(rotation)
    quat_scipy = rotations.as_quat()  # shape: (n_samples, 4)

    # Quaternions are in the following form w, x, y, z
    quaternion_estimated = np.column_stack(
        (quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2])
    )
    negative_mask = quaternion_estimated[0, 0] < 0
    quaternion_estimated[negative_mask] *= -1

    x_quaternion = quaternion_estimated[0, 1:4] / quaternion_estimated[0, 0]

    # Build the initial vector:
    # Start with two zeros, then the translation (as a row vector), then x_quaternion.

    X_init = np.concatenate(([0.0, 0.0], translation.flatten(), x_quaternion.flatten()))
    # X_init = np.concatenate(([0.0], [0.0]))

    # Append each 3D point (points is 3xN) column by column.
    # Flatten using Fortran order (column-major) to mimic MATLAB's behavior.
    aux = []
    for k in range(points.shape[1]):
        aux.append(points[0, k])
        aux.append(points[1, k])
        aux.append(points[2, k])
    points_new = np.array(aux)
    X_init = np.concatenate((X_init, points_new))

    return X_init


def init_optimization_pose(translation, rotation):
    # Computethe minimal rotation representation: x_quaternion = (x,y,z)/w.

    rotations = R.from_matrix(rotation)
    quat_scipy = rotations.as_quat()  # shape: (n_samples, 4)

    # Quaternions are in the following form w, x, y, z
    quaternion_estimated = np.column_stack(
        (quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2])
    )
    negative_mask = quaternion_estimated[0, 0] < 0
    quaternion_estimated[negative_mask] *= -1

    x_quaternion = quaternion_estimated[0, 1:4] / quaternion_estimated[0, 0]
    X_init = np.concatenate((translation.flatten(), x_quaternion.flatten()))
    return X_init


def cameraCalibrationPose(pts1, A, x_init, x):
    # Ensure pts1 and pts2 are of type float (double precision)
    pts1 = np.asarray(pts1, dtype=np.float64)

    # d = (size_optimization_x - 1) * size_optimization_y
    d = 6
    a_vector = ca.SX.sym("pse_estimation", d, 1)

    # Initialize cost to zero.
    cost = 0

    # Distortion coefficients: first 2 elements.
    # Translation: elements 3:5 (MATLAB indices 3:5 -> Python indices 2:5)
    x_trans = a_vector[0:3]
    # Rotation (minimal representation, e.g., a 3-vector): elements 6:8 (Python indices 5:8)
    x_quaternion = a_vector[3:6]

    # --- Map the 3-vector to a quaternion ---
    # Compute norm squared of x_quaternion.
    norm_sq = ca.dot(x_quaternion, x_quaternion)
    q0 = (1 - norm_sq) / (1 + norm_sq)
    q1 = 2 * x_quaternion[0] / (1 + norm_sq)
    q2 = 2 * x_quaternion[1] / (1 + norm_sq)
    q3 = 2 * x_quaternion[2] / (1 + norm_sq)
    quaternion = ca.vertcat(q0, q1, q2, q3)  # quaternion in form [w, x, y, z]
    R1 = quat_to_rot(quaternion)

    t1 = x_trans
    R1 = R1

    U_real1 = pts1[0:2, :]
    U_improved_final_1 = projection(R1, t1, A, x, pts1)

    ### --- Compute reprojection error ---
    ## U_real is a numpy array; convert it to CasADi DM.
    error_1 = U_real1 - U_improved_final_1
    ### Reshape error into a column vector.
    error_reshape_1 = ca.reshape(error_1, (2 * error_1.shape[1], 1))
    cost = cost + error_reshape_1.T @ error_reshape_1

    ### --- Set up and solve the NLP ---
    nlp = {"x": a_vector, "f": cost}
    opts = {"print_time": 1, "ipopt": {"print_level": 5}}
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    sol = solver(x0=x_init)
    a_opt_vec = np.array(sol["x"]).flatten()

    ### --- Extract optimized parameters ---
    ### Extract the optimized rotation and translation parts.
    x_quaternion_opt = a_opt_vec[3:6]
    norm_sq_opt = np.dot(x_quaternion_opt, x_quaternion_opt)
    q0_opt = (1 - norm_sq_opt) / (1 + norm_sq_opt)
    q1_opt = 2 * x_quaternion_opt[0] / (1 + norm_sq_opt)
    q2_opt = 2 * x_quaternion_opt[1] / (1 + norm_sq_opt)
    q3_opt = 2 * x_quaternion_opt[2] / (1 + norm_sq_opt)
    quaternion_opt = np.array([q0_opt, q1_opt, q2_opt, q3_opt])

    x_trans_opt = a_opt_vec[0:3]
    R_quaternion_opt = quat_to_rot(quaternion_opt)
    R_quaternion_opt = np.array(R_quaternion_opt)
    x_trans_opt = np.array(x_trans_opt)

    return x_trans_opt, R_quaternion_opt
