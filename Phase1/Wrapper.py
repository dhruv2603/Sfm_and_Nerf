import numpy as np
import os
import scipy.linalg
import tqdm
import argparse
from natsort import natsorted
import scipy
from scipy.optimize import least_squares
from EstimateFundamentalMatrix import (
    EstimateFundamentalMatrix,
    fundamental_analytical,
    FundamentalCasadi,
    fitRansac,
    recoverPoseFromFundamental,
    triangulatePoints,
    projection_values,
    init_optimization_variables,
    cameraCalibrationCasADi,
)
from GetInlierRANSAC import GetInlierRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from DisambiguateCameraPose import DisambiguateCameraPose
from helperFunctions import *
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from LinearTriangulation import LinearTriangulation

# Load the .mat file
data = sio.loadmat("data_simulation.mat")
# Access variables stored in the file


def SetData(dl, K):
    sz = len(dl)
    X = np.ones((3, sz))
    U = np.ones((3, sz))

    X_c = np.ones((3, sz))
    U_c = np.ones((3, sz))
    K_inv = np.linalg.inv(K)
    for k in range(sz):
        X[0, k] = float(dl[k][0])
        X[1, k] = float(dl[k][1])

        U[0, k] = float(dl[k][2])
        U[1, k] = float(dl[k][3])

        # Points respect to the center of the image
        X_c[:, k] = K_inv @ X[:, k]
        U_c[:, k] = K_inv @ U[:, k]
    return X, U, X_c, U_c


def main():
    # Define inputs of the Algorithm
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--Data",
        default="./P2Data",
        help="Path where the images are stored, Default:../Data/Train/Set2",
    )

    Args = Parser.parse_args()

    # Number of features for corner detector and possible Ransac
    DATA_DIR = Args.Data

    CALIBRATION_PATH, matching_files = sorttxtFiles(DATA_DIR)
    n = len(matching_files) + 1
    nFeatures, data_list = readFiles(matching_files, DATA_DIR)
    """
    Compute inliers using RANSAC
    """
    K = np.loadtxt(CALIBRATION_PATH)
    iteration = 0
    for img_n, dl in enumerate(data_list):
        if iteration == 0:
            # Get data as matlab
            uv_1, uv_2, uv_1_c, uv_2_c = SetData(dl, K)
            ## Initial fundamental matrix
            # F_init = fundamental_analytical(uv_1, uv_2)

            ## Compute Fundamental robust to oul;iers
            # best_model, num_iterations, inliers_index = fitRansac(
            #    uv_1, uv_2, 8, F_init, 0.001
            # )
            # U, s, Vh = np.linalg.svd(best_model)
            # vector = np.array([s[0], s[1], 0])

            F_rank, mask = cv2.findFundamentalMat(
                uv_1[0:2, :].T,
                uv_2[0:2, :].T,
                cv2.FM_RANSAC,
                ransacReprojThreshold=0.03,
                confidence=0.95,
                maxIters=5000,
            )
            # F_rank = U @ np.diag(vector) @ Vh
            print(np.linalg.matrix_rank(F_rank))
            inlier_indices = np.where(mask.ravel() == 1)[0]
            inliers_index = inlier_indices.tolist()

            # Get rotation and translation of the fundamental matrix
            R_ransac, t_ransac, _inliers = recoverPoseFromFundamental(
                F_rank, K, uv_1[0:2, :], uv_2[0:2, :]
            )
            # P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            # P2 = K @ np.hstack((R_ransac, t_ransac.reshape(3, 1)))

            I = np.eye(3, 3)
            t = np.zeros((3, 1))
            # H1 = np.block([[I, t], [np.zeros((1, 3)), np.array([[1]])]])
            # H2 = np.block(
            # [
            #    [R_ransac, t_ransac.reshape(3, 1)],
            #    [np.zeros((1, 3)), np.array([[1]])],
            # ]
            # )
            E = EssentialMatrixFromFundamentalMatrix(K, F_rank)
            R_l, C_l = ExtractCameraPose(E)

            # Projection on the 3d world

            pts3D_aux_0, aux_number_0 = LinearTriangulation(dl, K, R_l[0], (C_l[0]))
            pts3D_aux_0 = np.array(pts3D_aux_0).T
            print(aux_number_0)
            pts3D_aux_1, aux_number_1 = LinearTriangulation(dl, K, R_l[1], (C_l[1]))
            pts3D_aux_1 = np.array(pts3D_aux_1).T
            print(aux_number_1)
            pts3D_aux_2, aux_number_2 = LinearTriangulation(dl, K, R_l[2], C_l[2])
            pts3D_aux_2 = np.array(pts3D_aux_2).T
            print(aux_number_2)
            pts3D_aux_3, aux_number_3 = LinearTriangulation(dl, K, R_l[3], C_l[3])
            pts3D_aux_3 = np.array(pts3D_aux_3).T
            print(aux_number_3)

            P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = np.hstack((R_ransac, t_ransac.reshape(3, 1)))

            pts3D_4xN = triangulatePoints(uv_1, uv_2, P1, P2)
            pts3D_4xN = pts3D_4xN / pts3D_4xN[3, :]

            # Pixels values on the image
            # pixels_1 = projection_values(H1, pts3D_4xN, 0.0, 0.0, K)
            # pixels_2 = projection_values(H2, pts3D_4xN, 0.0, 0.0, K)

            # plotMatches(dl, n, img_n, DATA_DIR, pixels_1, pixels_2, "Classic")
            getMatchesNew(dl, inliers_index, n, img_n, DATA_DIR, "Ransacopencv")

            ### Optimization section
            x_init = init_optimization_variables(C_l[1], R_l[1], pts3D_aux_1[0:3, :])
            x_vector_opt, x_trans_opt, R_quaternion_opt, distortion_opt = (
                cameraCalibrationCasADi(
                    uv_1,
                    uv_2,
                    K,
                    x_init,
                    I,
                    t,
                    R_l[1],
                    C_l[1],
                    pts3D_aux_1,
                )
            )
            pts3D_4xN_casadi = np.vstack(
                (x_vector_opt, np.ones((1, x_vector_opt.shape[1])))
            )

            H3 = np.block(
                [
                    [R_quaternion_opt, x_trans_opt.reshape(3, 1)],
                    [np.zeros((1, 3)), np.array([[1]])],
                ]
            )

            #### Nonlinear projection
            # pixels_3 = projection_values(H1, pts3D_4xN_casadi, 0, 0, K)
            # pixels_4 = projection_values(H3, pts3D_4xN_casadi, 0, 0, K)

            # pixels_3 = np.array(pixels_3)
            # pixels_4 = np.array(pixels_4)

            # plotMatches(dl, n, img_n, DATA_DIR, pixels_3, pixels_4, "Non-linear")

            # Dhruv Method
            # inliers_dl, idxs = GetInlierRANSAC(dl)
            # getMatches(dl, inliers_index, n, img_n, DATA_DIR)
            # F = EstimateFundamentalMatrix(inliers_dl)
            print("Values")
        iteration = iteration + 1

        # Extract x, y, and z coordinates from the matrix
    plt.scatter(
        pts3D_aux_0[0, :], pts3D_aux_0[2, :], s=5, color="red", label="Dataset 0"
    )
    plt.scatter(
        pts3D_aux_1[0, :], pts3D_aux_1[2, :], s=5, color="green", label="Dataset 1"
    )
    plt.xlim(-20, 20)
    # plt.scatter(pts3D_4xN[0, :], pts3D_4xN[2, :], color="red", label="Dataset 3")
    # plt.scatter(
    # plt.scatter(
    # plt.scatter(
    plt.scatter(
        pts3D_4xN_casadi[0, :],
        pts3D_4xN_casadi[2, :],
        s=5,
        color="blue",
        label="Dataset 3",
    )

    # Labeling the axes and adding a title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Scatter Plot of Two Data Sets")
    plt.savefig("scatter_plot.pdf", format="pdf")


#

if __name__ == "__main__":
    main()
