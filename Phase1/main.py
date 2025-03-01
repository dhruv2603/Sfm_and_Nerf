import numpy as np
import argparse
import csv
from helperFunctions import *
import scipy.io as sio
import matplotlib.pyplot as plt
from GetInlierRANSAC import getFundamentalMatRANSAC
from EstimateFundamentalMatrix import getFundamentalMatrix
from ExtractCameraPose import recoverPoseFromFundamental
from LinearPnp import LinearPnP
from PnPRANSAC import PnPRANSAC
from LinearTriangulation import triangulatePoints, LinearTriangulation
import cv2 as cv2
from NonlinearTriangulation import init_optimization_variables, cameraCalibrationCasADi
from aux_functions import projection_values
from aux_functions import show_projection, show_projection_image


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

            # Tolerance
            tol = 1

            # Compute sift features from the images
            ptsA, ptsB = get_features(n, img_n, DATA_DIR)
            # plotMatches(dl, n, img_n, DATA_DIR, pixels_1, pixels_2, "Verification")

            # Compute fundamental matrix based on our functions
            F_aux = getFundamentalMatrix(pixels_1.T, pixels_2.T, num_point=8)
            F_aux, mask_sift = getFundamentalMatRANSAC(
                ptsA=pixels_1.T, ptsB=pixels_2.T, tol=tol, num_sample=8, confidence=0.99
            )

            # Compute fundamental matrix based on cv funciton
            F_ree, mask = cv2.findFundamentalMat(
                pixels_1.T,
                pixels_2.T,
                cv2.FM_RANSAC,
                ransacReprojThreshold=0.1,
                confidence=0.99,
                maxIters=5000,
            )

            # Get Re-estimate Fundamental matrix using only inliers
            inliersA_og = pixels_1.T[mask.ravel() == 1]
            inliersB_og = pixels_2.T[mask.ravel() == 1]
            inlier_indices = np.where(mask.ravel() == 1)[0]
            inliers_index = inlier_indices.tolist()

            ## Homogenous data shape 3, N
            points_A_normalized_inlier = np.vstack(
                (inliersA_og.T, np.ones((1, inliersA_og.shape[0])))
            )  # Shape: (3, N)
            points_B_normalized_inlier = np.vstack(
                (inliersB_og.T, np.ones((1, inliersB_og.shape[0])))
            )  # Shape: (3, N)

            ## Compute Rotation and translation
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

            ## Triangulation based on our funcitons
            pts3D_4xN = triangulatePoints(
                points_A_normalized_inlier, points_B_normalized_inlier, P1, P2
            )
            pts3D_4xN = pts3D_4xN / pts3D_4xN[3, :]

            ## Triangulation based on cv functions
            # pts3D_4xN = cv2.triangulatePoints(
            #    P1[0:3, 0:4],
            #    P2[0:3, 0:4],
            #    points_A_normalized_inlier[0:2, :],
            #    points_B_normalized_inlier[0:2, :],
            # )  # OpenCV's Linear-Eigen triangl
            # pts3D_4xN = pts3D_4xN / pts3D_4xN[3, :]

            # Nonlinear Optimizer for translations, rotation and points in world
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

            ## Points from the optimizer
            pts3D_4xN_casadi = np.vstack(
                (x_vector_opt, np.ones((1, x_vector_opt.shape[1])))
            )

            # Plot projection  nonlinear
            show_projection(
                x_trans_opt,
                R_quaternion_opt,
                pts3D_4xN_casadi,
                K,
                DATA_DIR,
                dl,
                n,
                img_n,
                inliers_index,
                "Non-linear",
            )

            show_projection(
                t_ransac,
                R_ransac,
                pts3D_4xN,
                K,
                DATA_DIR,
                dl,
                n,
                img_n,
                inliers_index,
                "linear",
            )
            ## Show results
            fig = plt.figure()

            # Add a 3D subplot
            ax = fig.add_subplot(111)
            plt.scatter(
                pts3D_4xN[0, :],
                pts3D_4xN[2, :],
                s=2,
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
            plt.xlim(-20, 20)
            plt.ylim(0, 30)
            # Labeling the axes and adding a title
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.title("2D Scatter Plot of Two Data Sets")
            plt.savefig("scatter_plot.pdf", format="pdf")

            print("R RANSAC: ", R_ransac)
            print("R Quaternion Opt: ", R_quaternion_opt)
            print("T RANSAC: ", t_ransac)
            print("X Trans Opt:", x_trans_opt)
            print("InliersA OG shape: ", inliersA_og.shape)
            print("InliersB OG shape: ", inliersA_og.shape)
            print("World Coordinates shape: ", pts3D_4xN_casadi.shape)
            print("Type A :", type(inliersA_og))
            print("Type B: ", type(inliersB_og))
            print("Type world: ", type(pts3D_4xN_casadi))
            # Array with values:
            # [World coordinate, img_id, u, v, img_id, u, v]
            # stacked one below the other for each point
            master_list = np.hstack(
                [
                    pts3D_4xN_casadi.T,
                    np.ones((inliersA_og.shape[0], 1), dtype=int),
                    inliersA_og,
                    2 * np.ones((inliersA_og.shape[0], 1), dtype=int),
                    inliersB_og,
                ]
            )
            master_list = master_list.tolist()
            print("Master List length: ", len(master_list))
            P = [P1, P2]
        iteration = iteration + 1

    # Aux variables data
    R_total = []
    t_total = []
    image_points = []
    world_points = []
    inliers_total = []

    # Traverse in the data list for each new image
    # for image i, get all pairs till i-1 (because you have world coordinates for i-1)
    for i in range(3, n + 1):
        # Store the world coordinates corresponding to each new image i (remember shape is features x 4)
        X_i = np.empty([0, 3])
        # store the corresponding image i pixels in another array
        x_i = np.empty([0, 2])
        # complete tringulation related to image i wrt all images j
        triangulate_j_list = []
        # for image i get images from 1 to i-1
        for j in range(1, i):
            # obtain the index of the data list match images (j,i)
            match_idx = (j - 1) * (10 - j) / 2 + i - j - 1
            # get the list matching[ji]
            print("Match idx", match_idx)
            dl = data_list[int(match_idx)]
            # get the uv indexes for j and i
            uv_j, uv_i, uv_j_c, uv_i_c = SetData(dl, K)
            # Perform RANSAC to remove outliers
            uv_j = uv_j.T[:, :2]
            uv_i = uv_i.T[:, :2]
            print("Shape uv_j", uv_j.shape)
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
        print(P_i)
        print(R_i)
        print(t_i)
        print(len(inlier_idxs))
        # P_i = LinearPnP(X_i,x_i,K)
        P.append(P_i)
        R_total.append(R_i)
        t_total.append(t_i)
        image_points.append(x_i)
        world_points.append(X_i)
        inliers_total.append(inlier_idxs)

        # Try to optimze based on the previous rotations and translations
        # x_init = init_optimization_variables(x_trans_opt, R_quaternion_opt, X_i)

        ## Uncomment the below lines
        # # Triangulate to get new world points
        # for j in range(1,i):
        #     # obtain the index of the data list match images (j,i)
        #     match_idx = (j-1)*(10-j)/2 + i-j-1
        #     # get the list matching[ji]
        #     print("Match idx", match_idx)
        #     dl = data_list[int(match_idx)]
        #     # get the uv indexes for j and i
        #     uv_j, uv_i, uv_j_c, uv_i_c = SetData(dl,K)
        #     # Perform RANSAC to remove outliers (need to implement)

        #     # Perform triangulation to get new world points
        #     print(triangulate_j_list[j-1])
        #     AA = uv_j[:,triangulate_j_list[j-1]]
        #     print(type(AA))
        #     print(AA)
        #     # BB = uv_i[:,triangulate_j_list[j-1]]
        #     pts3D_4xN = triangulatePoints(uv_j[:,triangulate_j_list[j-1]], uv_i[:,triangulate_j_list[j-1]], P[j-1], P[i-1])
        #     # Perform non-linear triangulation (need to implement)

        #     # Store the world points, img ids and img pixels in master list (need to implement)

    with open("./P2Data/Matches/master_list.txt", "w", newline="") as file:
        writer = csv.writer(file, delimiter=" ")
        # Write each list as a row
        writer.writerows(master_list)

    R_total = np.array(R_total)
    t_total = np.array(t_total)
    # print(R_total[0, :, :])
    print((t_total[0, :]))
    print(R_total[0, :, :].T @ (-t_total[0, :]))
    print(R_quaternion_opt.T @ (-x_trans_opt))
    # print(R_ransac.T @ (-t_ransac))

    world_points_3 = np.vstack(
        (world_points[0].T, np.ones((1, world_points[0].shape[0])))
    )

    # creating new points

    data_list_12 = np.vstack((np.array(data_list[1]), np.array(data_list[4]))).tolist()

    show_projection_image(
        t_total[0, :],
        R_total[0, :, :],
        world_points_3,
        K,
        DATA_DIR,
        data_list[1],
        n,
        img_n,
        inliers_total[0],
        "Linear 3",
        3,
    )

    show_projection_image(
        x_trans_opt,
        R_quaternion_opt,
        pts3D_4xN_casadi,
        K,
        DATA_DIR,
        data_list[0],
        n,
        img_n,
        inliers_index,
        "Nolinear 2",
        2,
    )


if __name__ == "__main__":
    main()
