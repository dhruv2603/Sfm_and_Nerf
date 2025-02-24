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
)
from GetInlierRANSAC import GetInlierRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from DisambiguateCameraPose import DisambiguateCameraPose
from helperFunctions import *
import scipy.io as sio

# Load the .mat file
data = sio.loadmat("data_simulation.mat")
# Access variables stored in the file


def SetData(dl):
    sz = len(dl)
    X = np.ones((3, sz))
    U = np.ones((3, sz))
    for k in range(sz):
        X[0, k] = float(dl[k][0])
        X[1, k] = float(dl[k][1])
        U[0, k] = float(dl[k][2])
        U[1, k] = float(dl[k][3])
    return X, U


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
    # U = data["data_uv"]
    # uv_1 = U[:, :, 0]
    # uv_2 = U[:, :, 1]
    # F = fundamental_analytical(uv_1, uv_2)
    # F_optimization = FundamentalCasadi(uv_1, uv_2, F)
    # best_model, num_iterations, inliers_index = fitRansac(uv_1, uv_2, 8, F, 0.01)
    # U, s, Vh = np.linalg.svd(best_model)
    # vector = np.array([s[0], s[1], 0])
    # F_rank = U @ np.diag(vector) @ Vh
    # K = 1.0e03 * np.array([[1.5000, 0, 0.6400], [0, 1.5000, 0.5120], [0, 0, 0.0010]])
    # R_ransac, t_ransac, _inliers = recoverPoseFromFundamental(
    #     F_rank, K, uv_1[0:2, :], uv_2[0:2, :]
    # )
    # print(F_rank)
    # print(R_ransac)
    # print(t_ransac)
    K = np.loadtxt(CALIBRATION_PATH)
    iteration = 0
    for img_n, dl in enumerate(data_list):
        if iteration == 0:
            uv_1, uv_2 = SetData(dl)
            F_init = fundamental_analytical(uv_1, uv_2)
            best_model, num_iterations, inliers_index = fitRansac(
                uv_1, uv_2, 8, F_init, 0.0001
            )
            U, s, Vh = np.linalg.svd(best_model)
            vector = np.array([s[0], s[1], 0])
            F_rank = U @ np.diag(vector) @ Vh

            R_ransac, t_ransac, _inliers = recoverPoseFromFundamental(
                F_rank, K, uv_1[0:2, :], uv_2[0:2, :]
            )

            t_ransac = R_ransac.T @ -t_ransac

            # Dhruv Method
            inliers_dl, idxs = GetInlierRANSAC(dl)
            getMatches(dl, inliers_index, n, img_n, DATA_DIR)
            F = EstimateFundamentalMatrix(inliers_dl)
            E = EssentialMatrixFromFundamentalMatrix(K, F)
            R_l, C_l = ExtractCameraPose(E)
            # R, C, world_coords = DisambiguateCameraPose(inliers_dl, K, R_l, C_l)

            # Verification
            print("Estimation of F matrix")
            print(F_rank)
            print(F)
            print("Rotation")
            print(R_ransac)
            print(R_l[0].T)
            print(R_l[1].T)
            print(R_l[2].T)
            print(R_l[3].T)
            print("Translation")
            print(t_ransac / t_ransac[2])
            print(C_l[0] / C_l[0][2])
            print(C_l[1] / C_l[1][2])
            print(C_l[2] / C_l[2][2])
            print(C_l[3] / C_l[3][2])
        iteration = iteration + 1


#

if __name__ == "__main__":
    main()
