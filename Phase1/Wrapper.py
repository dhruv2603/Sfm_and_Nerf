import numpy as np
import os
import tqdm
import argparse
from natsort import natsorted
from scipy.optimize import least_squares
from EstimateFundamentalMatrix import EstimateFundamentalMatrix
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
        
    """
    Estimate the Fundamental Matrix
    """
    nFeatures, data_list = readFiles(matching_files,DATA_DIR)
    for dl in data_list:
        F = EstimateFundamentalMatrix(DATA_DIR)
if __name__ == "__main__":
    main()