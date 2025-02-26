import numpy as np
import argparse
from helperFunctions import *
import scipy.io as sio
import matplotlib.pyplot as plt
from functions import *
from EstimateFundamentalMatrix import (
    recoverPoseFromFundamental,
    triangulatePoints,
    init_optimization_variables,
    cameraCalibrationCasADi,
)
import cv2 as cv2


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
            pixels_1 = uv_1[0:2, :]
            pixels_2 = uv_2[0:2, :]

            # Coputing Fundamental matrrix based on data
            F = getFundamentalMatrix(pixels_1.T, pixels_2.T, num_point=8)
            tol = 1

            # Ransac over the points
            # F_og, mask_og = getFundamentalMatRANSAC(
            # ptsA=pixels_1, ptsB=pixels_2, tol=tol, num_sample=8, confidence=0.99
            # )

            # sift over the pair of images to compute features
            ptsA, ptsB = get_features(n, img_n, DATA_DIR)
            plotMatches(dl, n, img_n, DATA_DIR, ptsA.T, ptsB.T, "Verification")

            # Compute fundamental matrix sift without ransacs
            F_aux = getFundamentalMatrix(ptsA, ptsB, num_point=8)

            F_aux, mask_sift = getFundamentalMatRANSAC(
                ptsA=ptsA, ptsB=ptsB, tol=tol, num_sample=8, confidence=0.99
            )

            # Get Re-estimate Fundamental matrix using only inliers
            inliersA_og = ptsA[mask_sift.ravel() == 1]
            inliersB_og = ptsB[mask_sift.ravel() == 1]
            F_ree, _ = getFundamentalMatRANSAC(
                ptsA=inliersA_og,
                ptsB=inliersB_og,
                tol=tol,
                num_sample=8,
                confidence=0.99,
            )
            ## Normalize data
            inliersA_og = np.vstack(
                (inliersA_og.T, np.ones((1, inliersA_og.shape[0])))
            )  # Shape: (3, N)
            inliersB_og = np.vstack(
                (inliersB_og.T, np.ones((1, inliersB_og.shape[0])))
            )  # Shape: (3, N)
            points_A_normalized_inlier = inliersA_og
            points_B_normalized_inlier = inliersB_og
            R_ransac, t_ransac, _inliers = recoverPoseFromFundamental(
                F_ree, K, points_A_normalized_inlier, points_B_normalized_inlier
            )
            ## Projection Matrix
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
            T_2 = np.vstack(
                (
                    np.hstack((R_ransac, t_ransac.reshape((3, 1)))),  # shape: (3,4)
                    aux_last_element_homogeneous,
                )
            )

            P1 = K @ F_identity @ Identity @ T_1
            P2 = K @ F_identity @ Identity @ T_2

            # pts3D_4xN = triangulatePoints(
            # points_A_normalized_inlier, points_B_normalized_inlier, P1, P2
            # )
            # pts3D_4xN = pts3D_4xN / pts3D_4xN[3, :]
            pts3D_4xN = cv2.triangulatePoints(
                P1[0:3, 0:4],
                P2[0:3, 0:4],
                points_A_normalized_inlier[0:2, :],
                points_B_normalized_inlier[0:2, :],
            )  # OpenCV's Linear-Eigen triangl
            pts3D_4xN = pts3D_4xN / pts3D_4xN[3, :]

            # Nonlinear Optimizer
            x_init = init_optimization_variables(t_ransac, R_ransac, pts3D_4xN[0:3, :])
            x_vector_opt, x_trans_opt, R_quaternion_opt, distortion_opt = (
                cameraCalibrationCasADi(
                    points_A_normalized_inlier,
                    points_B_normalized_inlier,
                    K,
                    x_init,
                    I,
                    t,
                    R_ransac,
                    t_ransac,
                    pts3D_4xN,
                )
            )
            pts3D_4xN_casadi = np.vstack(
                (x_vector_opt, np.ones((1, x_vector_opt.shape[1])))
            )
            plt.scatter(
                pts3D_4xN[0, :],
                pts3D_4xN[2, :],
                s=5,
                color="green",
                label="Dataset 3",
            )
            plt.scatter(
                pts3D_4xN_casadi[0, :],
                pts3D_4xN_casadi[2, :],
                s=1,
                color="blue",
                label="Dataset 3",
            )
            plt.xlim(-40, 40)

            # Labeling the axes and adding a title
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.title("2D Scatter Plot of Two Data Sets")
            plt.savefig("scatter_plot.pdf", format="pdf")

            print(R_ransac)
            print(R_quaternion_opt)
            print(t_ransac)
            print(x_trans_opt)

        iteration = iteration + 1


if __name__ == "__main__":
    main()
