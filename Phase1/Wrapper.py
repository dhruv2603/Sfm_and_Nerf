import numpy as np
import os
import scipy.linalg
import tqdm
import argparse
from natsort import natsorted
import scipy
from scipy.optimize import least_squares
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from GetInlierRANSAC import GetInlierRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from DisambiguateCameraPose import DisambiguateCameraPose
from helperFunctions import *


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


    # FOLDER = "../Data"
    # IMAGE_FOLDER = os.path.join(FOLDER, Args.Test)
    # IMAGE_DIR = os.path.join(IMAGE_FOLDER, Args.ImagesPath)

    # # Create folder to save the images
    # CORNERS_DIR = "./Corners/"
    # CORNERS_DIR = os.path.join(CORNERS_DIR, os.path.basename(IMAGE_DIR) + "/")

    # # Create new directory save as pdf
    # RESULTS_DIR = "./PDFs/"
    # RESULTS_DIR = os.path.join(RESULTS_DIR, os.path.basename(IMAGE_DIR) + "/")

    # # If not yet, create it
    # if not os.path.exists(CORNERS_DIR):
    #     os.makedirs(CORNERS_DIR)

    # if not os.path.exists(RESULTS_DIR):
    #     os.makedirs(RESULTS_DIR)

    CALIBRATION_PATH, matching_files = sorttxtFiles(DATA_DIR)
    n = len(matching_files) + 1
    nFeatures, data_list = readFiles(matching_files,DATA_DIR)
    """
    Compute inliers using RANSAC
    """
    for img_n, dl in enumerate(data_list):
        inliers_dl, idxs = GetInlierRANSAC(dl)
        getMatches(dl, idxs, n, img_n, DATA_DIR)
        """
        Estimate the Fundamental Matrix
        """
        F = EstimateFundamentalMatrix(inliers_dl)
        K = np.loadtxt(CALIBRATION_PATH)
        E = EssentialMatrixFromFundamentalMatrix(K, F)
        R_l,C_l = ExtractCameraPose(E)
        R,C, world_coords = DisambiguateCameraPose(inliers_dl, R_l, C_l)
if __name__ == "__main__":
    main()