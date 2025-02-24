import numpy as np
import os
import scipy
import scipy.linalg
import casadi as ca
import time
import math


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
                x1_h[0, i] * P1[2, :] - P1[0, :],
                x1_h[1, i] * P1[2, :] - P1[1, :],
                x2_h[0, i] * P2[2, :] - P2[0, :],
                x2_h[1, i] * P2[2, :] - P2[1, :],
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

    # Camera 1 projection matrix: [I | 0]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

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

            # Camera 2 projection matrix: [R | t]
            P2 = K @ np.hstack((R_test, t_test.reshape(3, 1)))

            # Triangulate points.
            pts3D_4xN = triangulatePoints(pts1_h, pts2_h, P1, P2)
            # Convert homogeneous 3D points to Cartesian coordinates.
            pts3D = pts3D_4xN[0:3, :] / pts3D_4xN[3, :]

            # Check depth in camera 1 (Z1 > 0).
            Z1 = pts3D[2, :]
            # Check depth in camera 2 (Z2 > 0): transform pts3D into camera 2 frame.
            pts3D_cam2 = R_test @ pts3D + t_test.reshape(3, 1)
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
