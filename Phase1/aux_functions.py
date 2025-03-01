import numpy as np
import os
import scipy
import scipy.linalg
import casadi as ca
import time
import math
from scipy.spatial.transform import Rotation as R
import cv2 as cv2
from matplotlib import pyplot as plt
from helperFunctions import plotMatches


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


def show_projection(
    x_trans_opt,
    R_quaternion_opt,
    pts3D_4xN_casadi,
    K,
    DATA_DIR,
    dl,
    n,
    img_n,
    inliers_index,
    name,
):

    # First image transformation
    I = np.eye(3, 3)
    t = np.zeros((3, 1))
    H1 = np.block([[I, t], [np.zeros((1, 3)), np.array([[1]])]])
    H3 = np.block(
        [
            [R_quaternion_opt, x_trans_opt.reshape(3, 1)],
            [np.zeros((1, 3)), np.array([[1]])],
        ]
    )

    # H1 = np.block([[I, t], [np.zeros((1, 3)), np.array([[1]])]])
    pixels_3 = projection_values(H1, pts3D_4xN_casadi, 0, 0, K)
    pixels_4 = projection_values(H3, pts3D_4xN_casadi, 0, 0, K)
    pixels_3 = np.array(pixels_3)
    pixels_4 = np.array(pixels_4)
    plotMatches(dl, inliers_index, n, img_n, DATA_DIR, pixels_3, pixels_4, name)


def show_projection_image(
    x_trans_opt,
    R_quaternion_opt,
    pts3D_4xN_casadi,
    K,
    DATA_DIR,
    dl,
    n,
    img_n,
    inliers_index,
    name,
    image_number,
):

    H3 = np.block(
        [
            [R_quaternion_opt, x_trans_opt.reshape(3, 1)],
            [np.zeros((1, 3)), np.array([[1]])],
        ]
    )

    pixels_4 = projection_values(H3, pts3D_4xN_casadi, 0, 0, K)
    pixels_4 = np.array(pixels_4)

    img2 = cv2.imread(os.path.join(DATA_DIR, str(image_number) + ".png"))
    color1 = (0, 0, 255)
    color2 = (255, 0, 0)

    # Plot points original
    for idx in range(len(dl)):
        pt2 = (int(float(dl[idx][2])), int(float(dl[idx][3])))
        cv2.circle(img2, pt2, 2, color1, -1)

    for k in range(0, pixels_4.shape[1]):
        projection_image_2 = (int(pixels_4[0, k]), int(pixels_4[1, k]))
        cv2.circle(img2, projection_image_2, 2, color2, -1)

    output_path = os.path.join(DATA_DIR, "image_projection")
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(output_path + "/" + name + "_" + "image_2" + ".png", img2)


def plotLinAndNonlinTri(
        C_opt,
        R_quaternion_opt,
        X_4xN_casadi,
        K,
        DATA_DIR, 
        data_list,
        n,
        inliers,
        R,
        C,
        X_4N):
    # Nonlinear Triangulation visualization
    show_projection(
        C_opt,
        R_quaternion_opt,
        X_4xN_casadi,
        K,
        DATA_DIR,
        data_list[0],
        n,
        0,
        inliers,
        "Non-linear",
    )

    # Linear Triangulation visualization
    show_projection(
        - R.T @ C,
        R,
        X_4N.T,
        K,
        DATA_DIR,
        data_list[0],
        n,
        0,
        inliers,
        "linear",
    )
    ## Show results
    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111)
    plt.scatter(
        X_4N.T[0, :],
        X_4N.T[2, :],
        s=2,
        color="green",
        label="Dataset 3",
    )
    plt.scatter(
        X_4xN_casadi[0, :],
        X_4xN_casadi[2, :],
        s=1,
        color="blue",
        label="Dataset 3",
    )
    plt.xlim(-20, 20)
    plt.ylim(0, 30)
    # Labeling the axes and adding a title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Scatter Plot of Two Data Sets")
    plt.savefig("scatter_plot.pdf", format="pdf")
