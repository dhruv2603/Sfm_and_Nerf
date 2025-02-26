import numpy as np
import os
import scipy
import scipy.linalg
import casadi as ca
import time
import math
from scipy.spatial.transform import Rotation as R
import cv2 as cv2


def EstimateFundamentalMatrix(dl):
    """
    This function calculates the fundamental matrix that represents the epipolar geometry between the camera poses.
    Input : List of u,v image pixel values for the two image views
    Output: The fundamental matrix F
    """
    A = np.empty([0, 9])
    for pt in dl:
        u = float(pt[0])
        v = float(pt[1])
        u_prime = float(pt[2])
        v_prime = float(pt[3])
        row = np.array(
            [
                u * u_prime,
                u * v_prime,
                u,
                v * u_prime,
                v * v_prime,
                v,
                u_prime,
                v_prime,
                1,
            ]
        )
        A = np.vstack((A, row))
    _, _, V = scipy.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    Uf, Df, Vf = scipy.linalg.svd(F)
    Df[-1] = 0
    F = np.matmul(Uf, np.matmul(np.diag(Df), Vf))
    return F


def fundamental_analytical(X, U):
    F_rows = []
    num_cols = U.shape[1]
    for k in range(num_cols):
        # Take first 3 rows from kth column of X and U
        x_col = X[:3, k]
        u_col = U[:3, k]
        # Compute the Kronecker product; np.kron returns a 1D array (length 9)
        aux_f = np.kron(x_col, u_col)
        F_rows.append(aux_f)
    F = np.array(F_rows)  # F will have shape (num_cols, 9)

    # Compute the SVD of F. Note: np.linalg.svd returns (U, s, Vh)
    U_f, S_f, Vh_f = np.linalg.svd(F, full_matrices=False)
    # Recover V (such that F = U_f * diag(S_f) * V^T)
    V_f = Vh_f.T

    # Find the index of the smallest singular value
    idx = np.argmin(S_f)
    # The corresponding singular vector from V
    F_nomr = V_f[:, idx]

    # Reshape the vector into a 3x3 matrix and transpose it
    F_norm = np.reshape(F_nomr, (3, 3)).T
    F_final = F_norm

    # Perform SVD on F_final to enforce a rank-2 constraint.
    U_final, S_final, Vh_final = np.linalg.svd(F_final)
    V_final = Vh_final.T  # Transpose to get V

    # Create a new diagonal matrix with S_final[0], S_final[1] and 0
    S_new = np.diag([S_final[0], S_final[1], 0])

    # Recompose the matrix with the reduced rank
    F_rank = U_final @ S_new @ V_final.T

    return F_rank


def FundamentalCasadi(X1, X2, x_init):
    # Create a symbolic variable x of dimension 9.
    x = ca.SX.sym("x", 9)

    # Build the matrix F by stacking the Kronecker products of the first three rows of each column of X1 and X2.
    F_rows = []
    num_cols = X1.shape[1]  # Assuming X1 and X2 have the same number of columns.
    for k in range(num_cols):
        # Extract first three elements of kth column from X1 and X2.
        x1_col = X1[:3, k]
        x2_col = X2[:3, k]
        # Compute the Kronecker product.
        aux_f = np.kron(x1_col, x2_col)
        # Append as a new row.
        F_rows.append(aux_f)
    F_np = np.vstack(F_rows)  # F_np is now a (num_cols x 9) matrix.

    # Convert the numeric matrix F_np into a CasADi DM (dense matrix).
    F_mat = ca.DM(F_np)

    # Define the residual r = F * x.
    r = F_mat @ x  # Matrix-vector multiplication.

    # Define the cost function: cost = r' * r.
    cost = ca.dot(r, r)

    # Define the constraint: g = x' * x.
    g = ca.dot(x, x)

    # Define the NLP (with g as the only constraint).
    nlp = {"x": x, "f": cost, "g": g}

    # Define solver options.
    opts = {"print_time": 0, "ipopt": {"print_level": 5}}

    # Create the IPOPT solver.
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Reshape x_init to be a column vector with shape (9, 1).
    x0 = np.reshape(x_init, (9, 1))

    # Time the solver.
    start_time = time.time()
    # Solve with the constraint lower and upper bounds both equal to 1.
    sol = solver(x0=x0, lbg=1, ubg=1)
    end_time = time.time()
    print("Solver time: {:.4f} seconds".format(end_time - start_time))

    # Extract the solution vector and reshape it into a 3x3 matrix.
    F_opt_vec = sol["x"]
    # Convert CasADi DM to a NumPy array and flatten.
    F_opt_np = ca.DM(F_opt_vec).full().flatten()
    F_opt = np.reshape(F_opt_np, (3, 3), order="F")

    return F_opt


def FundamentalCasadiAux(F, Y, x_init):
    # Ensure F and Y are in CasADi DM format for matrix operations.
    if not isinstance(F, ca.DM):
        F = ca.DM(F)
    if not isinstance(Y, ca.DM):
        Y = ca.DM(Y)

    # Define a symbolic variable x of size 9.
    x = ca.SX.sym("x", 9)

    # Define the residual: r = F*x - Y
    r = F @ x - Y
    # Define the cost function as the squared norm of r.
    cost = ca.dot(r, r)
    # Define the constraint: g = x'*x.
    g = ca.dot(x, x)

    # Define the nonlinear programming problem.
    nlp = {"x": x, "f": cost, "g": g}

    # Set IPOPT options.
    opts = {"print_time": 0, "ipopt": {"print_level": 0}}

    # Create the solver.
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Reshape the initial guess into a column vector of shape (9, 1).
    x0 = np.reshape(x_init, (9, 1))

    # Time the solver execution.
    start_time = time.time()
    sol = solver(x0=x0, lbg=1, ubg=1)
    end_time = time.time()
    print("Solver time: {:.4f} seconds".format(end_time - start_time))

    # Extract the solution vector and reshape it into a 3x3 matrix.
    F_opt_vec = sol["x"]
    F_opt_np = ca.DM(F_opt_vec).full().flatten()
    F_opt = np.reshape(F_opt_np, (3, 3), order="F")

    return F_opt


def fitRansac(X1, X2, num_sample, F_init, threshold):
    # Initialization
    num_iterations = np.inf
    iterations_done = 0
    max_inlier_count = 0
    best_model = None

    desired_prob = 0.95

    # Build F and Y from the input columns.
    # Each row of F is the Kronecker product of the first 3 elements (rows) of the corresponding column in X1 and X2.
    num_points = X1.shape[1]
    F_list = []
    Y_list = []

    for k in range(num_points):
        # Get first three elements from the k-th column
        aux_f = np.kron(X1[:3, k], X2[:3, k])  # shape: (9,)
        F_list.append(aux_f)
        # Y is set to zero for each sample
        Y_list.append(0)

    F = np.vstack(F_list)  # F becomes an (N x 9) matrix.
    Y = np.array(Y_list).reshape(-1, 1)  # Y becomes a column vector of size (N x 1).

    # Combine F and Y for easier shuffling: each row is [F_row, Y_value]
    total_data = np.hstack([F, Y])  # shape: (N x 10)
    data_size = total_data.shape[0]

    # RANSAC loop (adaptive number of iterations)
    while num_iterations > iterations_done:
        # Shuffle the rows of total_data
        idx = np.random.permutation(data_size)
        total_data_shuffled = total_data[idx, :]

        # Select the first 'num_sample' rows as the sample subset
        sample_data = total_data_shuffled[:num_sample, :]
        F_samp = sample_data[:, :-1]  # All columns except the last (F part)
        Y_samp = sample_data[:, -1]  # Last column (Y values)

        # Fit model to the sample subset using FundamentalCasadiAux
        F_optimization = FundamentalCasadiAux(F_samp, Y_samp, F_init)

        # Compute error across all data:
        # Flatten the optimized model into a (9 x 1) vector.
        estimated_model = np.array(
            [
                [float(F_optimization[0, 0])],
                [float(F_optimization[1, 0])],
                [float(F_optimization[2, 0])],
                [float(F_optimization[0, 1])],
                [float(F_optimization[1, 1])],
                [float(F_optimization[2, 1])],
                [float(F_optimization[0, 2])],
                [float(F_optimization[1, 2])],
                [float(F_optimization[2, 2])],
            ]
        )
        # Predicted Y values for all data
        y_estimated = F @ estimated_model  # (N x 1)
        err = np.abs(Y - y_estimated)

        # Count the number of inliers (points with error below the threshold)
        inlier_count = np.sum(err < threshold)

        # Update the best model if this model has more inliers than previous ones
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            best_model = F_optimization

        # Update the estimated number of iterations needed:
        prob_outlier = 1 - (inlier_count / data_size)
        denom = np.log(1 - (1 - prob_outlier) ** num_sample)
        if np.isclose(denom, 0):
            num_iterations = 0
        else:
            num_iterations = np.log(1 - desired_prob) / denom
        iterations_done += 1

    # Recompute the predicted Y for all data points using the previously computed F.
    y_estimated = F @ estimated_model  # F was constructed earlier from your data

    # Compute the absolute error between predicted Y and the actual Y.
    err = np.abs(Y - y_estimated)

    inlier_indices = np.where(err < threshold)[0]
    print("Inliear Found")
    print(inlier_indices.shape)

    return best_model, iterations_done, inlier_indices.tolist()


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


def recoverPoseFromFundamental(F, K, pts1, pts2):
    # Compute the essential matrix: E = K'.F.K
    E = K.T @ F @ K

    # SVD of E
    U, S, Vt = np.linalg.svd(E)
    # Force the singular values to be [1, 1, 0]
    S_new = np.array([1, 1, 0])
    E = U @ np.diag(S_new) @ Vt

    # Re-decompose E after enforcing the singular values
    U, _, Vt = np.linalg.svd(E)

    # Define W
    W = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    # Two candidate rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Candidate translation vectors (the 3rd column of U, with +/- sign)
    u3 = U[:, 2]
    t_candidates = [u3, -u3]
    R_candidates = [R1, R2]

    # Convert input points to homogeneous coordinates for triangulation.
    # pts1 and pts2 are assumed to be (N, 2) arrays.
    N = pts1.shape[0]
    pts1_h = np.vstack((pts1.T, np.ones((1, N))))  # Shape: (3, N)
    pts2_h = np.vstack((pts2.T, np.ones((1, N))))  # Shape: (3, N)

    F_identity = np.eye(3)
    Identity = np.hstack([np.eye(3), np.zeros((3, 1))])  # 3x4 matri
    I = np.eye(3, 3)
    t = np.zeros((3, 1))
    aux_last_element_homogeneous = np.array([[0.0, 0.0, 0.0, 1.0]])
    T_1 = np.vstack(
        (
            np.hstack((I, t)),  # shape: (3,4)
            aux_last_element_homogeneous,
        )
    )

    P1 = K @ F_identity @ Identity @ T_1

    bestCount = -np.inf
    best_R = None
    best_t = None
    best_inliers = None

    # Evaluate all 4 combinations (2 rotations x 2 translations) via cheirality check.
    for R_test in R_candidates:
        for t_test in t_candidates:
            # Ensure that the rotation has a positive determinant.
            if np.linalg.det(R_test) < 0:
                t_test = -t_test
                R_test = -R_test

            T_2 = np.vstack(
                (
                    np.hstack((R_test, t_test.reshape((3, 1)))),  # shape: (3,4)
                    aux_last_element_homogeneous,
                )
            )
            # Camera 2 projection matrix: [R | t]
            P2 = K @ F_identity @ Identity @ T_2

            # Triangulate points.
            pts3D = cv2.triangulatePoints(
                P1[0:3, 0:4],
                P2[0:3, 0:4],
                pts1_h[0:2, :],
                pts2_h[0:2, :],
            )  # OpenCV's Linear-Eigen triangl
            pts3D = pts3D / pts3D[3, :]

            # Check depth in camera 1 (Z1 > 0).
            Z1 = pts3D[2, :]
            # Check depth in camera 2 (Z2 > 0): transform pts3D into camera 2 frame.
            pts3D_cam2 = R_test @ pts3D[0:3, :] + t_test.reshape(3, 1)
            Z2 = pts3D_cam2[2, :]

            valid = (Z1 > 0) & (Z2 > 0)
            numInFront = np.sum(valid)

            if numInFront > bestCount:
                bestCount = numInFront
                best_R = R_test
                best_t = t_test
                best_inliers = valid

    R = best_R
    t = best_t
    inliers = best_inliers
    return R, t, inliers


def projection_values(H1, pts3D_4xN, k1, k2, K):
    # Define the identity matrices used in the transformation.
    F_identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    Identity = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    # Compute the normalized image coordinates.
    # Note: F_identity @ Identity is just Identity.
    values_normalized_estimated = F_identity @ Identity @ H1 @ pts3D_4xN
    # Divide the first two rows by the third row (perspective division)
    values_normalized_estimated = (
        values_normalized_estimated[:2, :] / values_normalized_estimated[2, :]
    )

    # Compute the radial distance for each point (Euclidean norm along axis 0).
    radius_estimated = np.linalg.norm(values_normalized_estimated, axis=0)

    # Compute the distortion factor.
    D_estimated = 1 + k1 * (radius_estimated**2) + k2 * (radius_estimated**4)

    # Apply the distortion (multiply each column by its corresponding distortion factor)
    x_warp_estimated = values_normalized_estimated * D_estimated
    # Append a row of ones to convert back to homogeneous coordinates.
    ones_row = np.ones((1, x_warp_estimated.shape[1]))
    x_warp_estimated = np.vstack((x_warp_estimated, ones_row))

    # Project the distorted coordinates using the camera intrinsic matrix.
    pixels_aux_estimated = K @ x_warp_estimated
    # Convert from homogeneous coordinates to 2D by dividing by the third row.
    pixels_aux_estimated = pixels_aux_estimated[:2, :] / pixels_aux_estimated[2, :]

    return pixels_aux_estimated


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
    print(U_real1.shape)
    print(U_improved_final_1.shape)
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
    opts = {"print_time": 1, "ipopt": {"print_level": 5}}
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
