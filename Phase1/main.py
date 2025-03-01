import numpy as np
import argparse
import csv
from helperFunctions import *
import scipy.io as sio
import matplotlib.pyplot as plt
from GetInlierRANSAC import getFundamentalMatRANSAC, GetInlierRANSAC
from EstimateFundamentalMatrix import getFundamentalMatrix, EstimateFundamentalMatrix
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose  # ,recoverPoseFromFundamental
from LinearTriangulation import triangulatePoints, LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from NonlinearTriangulation import (
    init_optimization_variables,
    cameraCalibrationCasADi,
    init_optimization_pose,
    cameraCalibrationPose,
)
from aux_functions import show_projection, show_projection_image
from PnPRANSAC import PnPRANSAC
from plot_results import plot_3d_results

# from LinearPnp import LinearPnP
# from PnPRANSAC import PnPRANSAC
# import cv2 as cv2
# from aux_functions import projection_values


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
    # Path for calibration matrix and list of matching_ij.txt filenames
    CALIBRATION_PATH, matching_files = sorttxtFiles(DATA_DIR)
    n = len(matching_files) + 1
    # Create a list of lists of all the matching pairs[[12],[13],[14] ... [45]]
    nFeatures, data_list = readFiles(matching_files, DATA_DIR)
    # Load the intrinsic calibration matrix
    K = np.loadtxt(CALIBRATION_PATH)
    # Get the pixel values for the img 1 and img 2
    # Shape of each is (3,N)
    uv_1, uv_2, uv_1_c, uv_2_c = SetData(data_list[0], K)

    """
    Compute inliers using RANSAC
    """
    homography_inliers = homography_RANSAC(uv_1[:2, :].T, uv_2[:2, :].T)
    print(uv_1.shape[1])
    print("Number of inliers from Homography RANSAC: ", len(homography_inliers))

    inliers, num_inliers = GetInlierRANSAC(
        uv_1[:2, :].T, uv_2[:2, :].T, homography_inliers
    )
    print("Number of inliers from 8-pt RANSAC: ", num_inliers)
    getMatches(data_list[0], inliers, n, 0, DATA_DIR)

    """
    Estimate the Fundamental Matrix
    """
    F = EstimateFundamentalMatrix(uv_1[:2, :].T[inliers], uv_2[:2, :].T[inliers])
    print("Fundamental Matrix: ", F)
    F_cv, mask = cv2.findFundamentalMat(
        uv_1[:2, :].T[inliers], uv_2[:2, :].T[inliers], method=cv2.FM_8POINT
    )
    print("Fundamental matrix from cv2", F_cv)

    e1, e2 = get_epipoles(F)
    print("Epipoles: ", e1, e2)

    # Get Epipolar Lines
    lines1, lines2 = get_epipolar_lines(
        F, uv_1[:2, :].T[inliers], uv_2[:2, :].T[inliers]
    )
    print("Epipolar Lines 1: ", lines1)
    print("Epipolar Lines 2: ", lines2)

    img1 = cv2.imread(os.path.join(DATA_DIR, "1.png"))
    img2 = cv2.imread(os.path.join(DATA_DIR, "2.png"))
    # Draw the epipolar lines
    img1_ep, img2_ep = drawlines(
        img1.copy(),
        img2.copy(),
        lines1,
        uv_1[:2, :].T[inliers],
        uv_2[:2, :].T[inliers],
        DATA_DIR,
    )
    # Draw the epipolar lines
    img1_ep_hat, img2_ep_hat = drawlines(
        img2.copy(),
        img1.copy(),
        lines2,
        uv_2[:2, :].T[inliers],
        uv_1[:2, :].T[inliers],
        DATA_DIR,
    )

    path = os.path.join(DATA_DIR, "1_epipoles.png")
    cv2.imwrite(path, img1_ep)
    path = os.path.join(DATA_DIR, "2_epipoles.png")
    cv2.imwrite(path, img2_ep)
    path = os.path.join(DATA_DIR, "1_epipoles_hat.png")
    cv2.imwrite(path, img1_ep_hat)
    path = os.path.join(DATA_DIR, "2_epipoles_hat.png")
    cv2.imwrite(path, img2_ep_hat)

    """
    Estimate Essential Matrix
    """
    E = EssentialMatrixFromFundamentalMatrix(K, F)
    print("Essential Matrix: ", E)

    """
    Perform Linear Triangulation
    """
    camera_poses = ExtractCameraPose(E)

    C0 = np.zeros(3)
    R0 = np.eye(3)
    X_4_comb = []
    C_4_list = []
    R_4_list = []

    for C, R in camera_poses:
        X = LinearTriangulation(
            K, C0, R0, C, R, uv_1[:2, :].T[inliers], uv_2[:2, :].T[inliers]
        )
        X_4_comb.append(X)
        C_4_list.append(C)
        R_4_list.append(R)

    """
    Disambiguate Camera Pose
    """
    C, R, X = DisambiguateCameraPose(R_4_list, C_4_list, X_4_comb)
    C = np.array(C)
    R = np.array(R)
    print("Rotation Matrix: ", R)
    print("Camera position: ", C)
    X = np.array(X)
    X_4N = np.hstack((X, np.ones((X.shape[0], 1))))

    """
    Perform Non Linear Triangulation
    """
    # Nonlinear Optimizer for translations, rotation and points in world
    # Initial values
    x_init = init_optimization_variables(C, R, X.T)
    # Points from the optimizer
    X_opt, C_opt, R_quaternion_opt, distortion_opt = cameraCalibrationCasADi(
        uv_1.T[inliers].T,
        uv_2.T[inliers].T,
        K,
        x_init,
        R0,
        C0.reshape((3, 1)),
        R,
        C,
        X_4N.T,
    )
    # Homogenization
    X_4xN_casadi = np.vstack((X_opt, np.ones((1, X_opt.shape[1]))))

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
        -R.T @ C,
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
    # Labeling the axes and adding a title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Scatter Plot of Two Data Sets")
    plt.savefig("scatter_plot.pdf", format="pdf")

    aux = uv_1[:2, :].T[inliers]
    # Array with values:
    # [World coordinate, img_id, u, v, img_id, u, v]
    # stacked one below the other for each point
    master_list = np.hstack(
        [
            X_4xN_casadi.T,
            np.ones((X_4xN_casadi.shape[1], 1), dtype=int),
            uv_1[:2, :].T[inliers],
            2 * np.ones((X_4xN_casadi.shape[1], 1), dtype=int),
            uv_2[:2, :].T[inliers],
        ]
    )
    master_list = master_list.tolist()

    # Aux variables data
    R_total = []
    t_total = []
    image_points = []
    world_points = []
    inliers_total = []

    # Init Orientations and translation for the optimizer
    translation_init = C_opt
    rotation_init = R_quaternion_opt

    tranlation_total = []
    orientation_total = []
    # save poses
    tranlation_total.append(C_opt)
    orientation_total.append(R_quaternion_opt)
    X_world_points = np.empty([0, 4])

    # Traverse in the data list for each new image
    # for image i, get all pairs till i-1 (because you have world coordinates for i-1)
    for i in range(3, n + 1):
        # Store the world coordinates corresponding to each new image i (remember shape is features x 4)
        X_i = np.empty([0, 3])
        # store the corresponding image i pixels in another array
        x_i = np.empty([0, 2])
        x_j = np.empty([0, 2])
        # complete tringulation related to image i wrt all images j
        triangulate_j_list = []
        # for image i get images from 1 to i-1
        for j in range(1, i):
            # obtain the index of the data list match images (j,i)
            match_idx = (j - 1) * (10 - j) / 2 + i - j - 1
            # get the list matching[ji]
            dl = data_list[int(match_idx)]
            # get the uv indexes for j and i
            uv_j, uv_i, uv_j_c, uv_i_c = SetData(dl, K)
            # Perform RANSAC to remove outliers
            uv_j = uv_j.T[:, :2]
            uv_i = uv_i.T[:, :2]
            # store indexes of array which need triangulation
            needs_triangulation_idxs_list = []
            # for each row in uv_j
            for a, each_row in enumerate(uv_j):
                # Flag to check if the point is already added in the master list
                flag_a_in_ml = 0
                # and each row in Master list
                for each_Mrow in master_list:
                    # calculate the length of the Master list row
                    Mrow_len = len(each_Mrow)
                    k = 0
                    # traverse throgh all ids in the row and check if the row has the id j
                    while 3 + 3 * k + 1 < Mrow_len:
                        if each_Mrow[3 + 3 * k + 1] == j:
                            if (
                                each_Mrow[3 + 3 * k + 2] == each_row[0]
                                and each_Mrow[3 + 3 * k + 3] == each_row[1]
                            ):
                                flag_a_in_ml = 1
                                m = k + 1
                                flag = 0
                                while 3 + 3 * m + 1 < Mrow_len:
                                    if each_Mrow[3 + 3 * k + 1] == j:
                                        flag = 1
                                        break
                                    m = m + 1
                                if flag == 1:
                                    break
                                each_Mrow.append(i)
                                each_Mrow.append(uv_i[a, 0])
                                each_Mrow.append(uv_i[a, 1])
                                X_i = np.vstack([X_i, each_Mrow[:3]])
                                x_i = np.vstack([x_i, uv_i[a]])
                                x_j = np.vstack([x_j, uv_j[a]])
                                break
                        k = k + 1
                if flag_a_in_ml == 0:
                    # store the index list in the matching list for which there is no world point
                    needs_triangulation_idxs_list.append(a)
            # print("Needs Triangulation len: ", len(needs_triangulation_idxs_list))
            # print("Needs triangulation list: ", needs_triangulation_idxs_list)
            triangulate_j_list.append(needs_triangulation_idxs_list)

        # Calculate the P matrix
        P_i, inlier_idxs, R_i, t_i = PnPRANSAC(X_i, x_i, K)
        R_total.append(R_i)
        t_total.append(t_i)
        image_points.append(x_i)
        world_points.append(X_i)
        inliers_total.append(inlier_idxs)

        # Nonlinear tirangulation between images
        world_points_data = np.vstack(
            (X_i[inlier_idxs, :].T, np.ones((1, X_i[inlier_idxs, :].shape[0])))
        )
        ## initial Condition
        x_init = init_optimization_pose(translation_init, rotation_init)

        # Optimization problem
        t_new, R_new = cameraCalibrationPose(
            x_i[inlier_idxs, :].T, K, x_init, world_points_data[0:3, :]
        )
        P1 = K @ np.hstack((rotation_init, translation_init.reshape(3, 1)))
        P2 = K @ np.hstack((R_new, t_new.reshape(3, 1)))

        X = triangulatePoints(x_j[inlier_idxs, :].T, x_i[inlier_idxs, :].T, P1, P2)
        X = X / X[3, :]

        x_init = init_optimization_variables(t_new, R_new, X)
        X_opt, C_opt, R_quaternion_opt, distortion_opt = cameraCalibrationCasADi(
            x_j[inlier_idxs, :].T,
            x_i[inlier_idxs, :].T,
            K,
            x_init,
            rotation_init,
            translation_init,
            R_new,
            t_new,
            X,
        )
        # Homogenization
        X_4xN_casadi = np.vstack((X_opt, np.ones((1, X_opt.shape[1]))))

        ## Computing triangulation
        X_world_points = np.vstack([X_world_points, X_4xN_casadi.T])

        # Saving data
        tranlation_total.append(t_new)
        orientation_total.append(R_new)

        # Set initials
        translation_init = t_new
        rotation_init = R_new

    print(-orientation_total[0].T @ tranlation_total[0])
    print(-orientation_total[1].T @ tranlation_total[1])

    with open("./P2Data/Matches/master_list.txt", "w", newline="") as file:
        writer = csv.writer(file, delimiter=" ")
        # Write each list as a row
        writer.writerows(master_list)
    plot_3d_results(tranlation_total, orientation_total, X_4xN_casadi, X_world_points.T)


if __name__ == "__main__":
    main()
