import numpy as np
import os
import scipy.linalg
from tqdm import tqdm
import argparse
from natsort import natsorted
import scipy
import cv2
from scipy.optimize import least_squares
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
from GetInlierRANSAC import GetInlierRANSAC
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from DisambiguateCameraPose import DisambiguateCameraPose
from NonLinearTriangulation import NonLinearTriangulation
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

    # IMAGE_FOLDER = os.path.join(FOLDER, Args.Test)
    # IMAGE_DIR = os.path.join(IMAGE_FOLDER, Args.ImagesPath)

    # # Create new directory save as pdf
    # RESULTS_DIR = "./PDFs/"
    # RESULTS_DIR = os.path.join(RESULTS_DIR, os.path.basename(IMAGE_DIR) + "/")

    # # If not yet, create it
    # if not os.path.exists(CORNERS_DIR):
    #     os.makedirs(CORNERS_DIR)

    CALIBRATION_PATH, matching_files = sorttxtFiles(DATA_DIR)
    n = len(matching_files) + 1
    nFeatures, data_list = readFiles(matching_files,DATA_DIR)
    for img_n, dl in enumerate(data_list):
        dl_array = np.array(dl)
        pts1 = dl_array[:,:2]
        pts2 = dl_array[:,2:]
        """
        Compute inliers using RANSAC
        """
        """
        Estimate the Fundamental Matrix
        """
        # inliers_dl, idxs = GetInlierRANSAC(pts1,pts2)
        F,idxs = GetInlierRANSAC(pts1,pts2)
        # print(F)
        F2,mask = cv2.findFundamentalMat(pts1, pts2) 
        # print(F2)
        getMatches(pts1,pts2, idxs, n, img_n, DATA_DIR)
        # i,j = getImgNums(n,img_n)
        # img1 = cv2.imread(os.path.join(DATA_DIR,str(i) + ".png"))
        # img2 = cv2.imread(os.path.join(DATA_DIR,str(j) + ".png"))
        # visualize_epipolar_lines(img1, img2, pts1, pts2, F, title="Epipolar Lines")
        # F = EstimateFundamentalMatrix(pts1,pts2)
        # if img_n == 0:
        #     F = EstimateFundamentalMatrix(pts1,pts2)

    # """
    # Estimamte the Intrinsic Matrix
    # """
    # K = np.loadtxt(CALIBRATION_PATH)
    # """
    # Estimate the Essential Matrix
    # """
    # E = EssentialMatrixFromFundamentalMatrix(K, F)
    # """
    # Extract Pose from Essential Matrix
    # """
    # R_l,C_l = ExtractCameraPose(E)
    # """
    # Get Initial World Coordinates using Linear Triangulation
    # """
    # R,C, X0 = DisambiguateCameraPose(inliers_dl, K, R_l, C_l)
    # """
    # Get Corrected World Coordinates using Non-Linear Triangualtion
    # """
    # X = NonLinearTriangulation(inliers_dl, K, R, C, X0)
    # # print(len(X))
    # """
    # Perform PnP RANSAC
    # """
    # print(R)
    # print(C)
if __name__ == "__main__":
    main()