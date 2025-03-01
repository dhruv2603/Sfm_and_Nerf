import numpy as np
import argparse
import csv
from helperFunctions import *
import scipy.io as sio
import matplotlib.pyplot as plt
from GetInlierRANSAC import getFundamentalMatRANSAC,GetInlierRANSAC
from EstimateFundamentalMatrix import getFundamentalMatrix, EstimateFundamentalMatrix
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose #,recoverPoseFromFundamental
from LinearTriangulation import triangulatePoints, LinearTriangulation
from DisambiguateCameraPose import DisambiguateCameraPose
from NonlinearTriangulation import init_optimization_variables, cameraCalibrationCasADi, init_optimization_pose, cameraCalibrationPose
from aux_functions import show_projection, show_projection_image, plotLinAndNonlinTri
from PnPRANSAC import PnPRANSAC
# from LinearPnp import LinearPnP
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
    homography_inliers = homography_RANSAC(uv_1[:2,:].T, uv_2[:2,:].T)
    print("Total number of features: ", uv_1.shape[1])
    print("Number of inliers from Homography RANSAC: ",len(homography_inliers))

    inliers, num_inliers = GetInlierRANSAC(uv_1[:2,:].T,uv_2[:2,:].T,homography_inliers)
    print("Number of inliers from 8-pt RANSAC: ", num_inliers)
    getMatches(data_list[0], inliers, n, 0, DATA_DIR)

    """
    Estimate the Fundamental Matrix
    """
    F = EstimateFundamentalMatrix(uv_1[:2,:].T[inliers], uv_2[:2,:].T[inliers])
    print("Fundamental Matrix: ", F)
    F_cv, mask = cv2.findFundamentalMat(uv_1[:2,:].T[inliers], uv_2[:2,:].T[inliers], method=cv2.FM_8POINT)
    print("Fundamental matrix from cv2", F_cv)

    e1, e2 = get_epipoles(F)
    print("Epipoles: ", e1, e2)
    
    # Get Epipolar Lines
    lines1, lines2 = get_epipolar_lines(F, uv_1[:2,:].T[inliers], uv_2[:2,:].T[inliers])
    
    img1 = cv2.imread(os.path.join(DATA_DIR,"1.png"))
    img2 = cv2.imread(os.path.join(DATA_DIR,"2.png"))
    # Draw the epipolar lines
    img1_ep, img2_ep = drawlines(img1.copy(), img2.copy(), lines1, uv_1[:2,:].T[inliers], uv_2[:2,:].T[inliers],DATA_DIR)
    # Draw the epipolar lines
    img1_ep_hat, img2_ep_hat = drawlines(img2.copy(), img1.copy(), lines2, uv_2[:2,:].T[inliers], uv_1[:2,:].T[inliers],DATA_DIR)
    
    path = os.path.join(DATA_DIR,"1_epipoles.png")
    cv2.imwrite(path,img1_ep)
    path = os.path.join(DATA_DIR,"2_epipoles.png")
    cv2.imwrite(path,img2_ep)
    path = os.path.join(DATA_DIR,"1_epipoles_hat.png")
    cv2.imwrite(path,img1_ep_hat)
    path = os.path.join(DATA_DIR,"2_epipoles_hat.png")
    cv2.imwrite(path,img2_ep_hat)

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
    
    for (C,R) in camera_poses:
        X = LinearTriangulation(K,C0,R0,C,R,uv_1[:2,:].T[inliers], uv_2[:2,:].T[inliers])
        X_4_comb.append(X)
        C_4_list.append(C)
        R_4_list.append(R)
    
    """
    Disambiguate Camera Pose
    """
    C, R, X = DisambiguateCameraPose(R_4_list, C_4_list, X_4_comb)
    C = np.array(C)
    R = np.array(R)
    print("Rotation Matrix: ",R)
    print("Camera position: ",C)
    X    = np.array(X)
    X_4N = np.hstack((X,np.ones((X.shape[0],1))))

    """
    Perform Non Linear Triangulation
    """
    # Nonlinear Optimizer for translations, rotation and points in world
    # Initial values
    x_init = init_optimization_variables(C, R, X.T)
    # Points from the optimizer
    X_opt, C_opt, R_quaternion_opt, distortion_opt = (
        cameraCalibrationCasADi(
            uv_1.T[inliers].T,
            uv_2.T[inliers].T,
            K,
            x_init,
            R0,
            C0.reshape((3,1)),
            R,
            C,
            X_4N.T,
        )
    )
    # Homogenization
    X_4xN_casadi = np.vstack(
        (X_opt, np.ones((1, X_opt.shape[1])))
    )

    # Plot the Linear porjecctions and non-linear projections
    plotLinAndNonlinTri(
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
        X_4N)

    # Array with values:
    # [World coordinate, img_id, u, v, img_id, u, v]
    # stacked one below the other for each point
    master_list = np.hstack(
        [
            X_4xN_casadi.T,
            np.ones((uv_1.T[inliers].shape[0], 1), dtype=int),
            uv_1[:2,:].T[inliers],
            2 * np.ones((uv_1.T[inliers].shape[0], 1), dtype=int),
            uv_2[:2,:].T[inliers],
        ]
    )
    print("Master list shape: ",master_list.shape)
    master_list = master_list.tolist()
    print("Master List length: ", len(master_list))
    print("R optimized for non-linear: ", R_quaternion_opt)
    R_list = [R0,R_quaternion_opt]
    C_list = [C0, C_opt]
    
    # Aux variables data
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
            # get the list matching[ji]
            match_idx = get_idx(i,j)
            print("Match idx", match_idx)
            dl = data_list[int(match_idx)]
            # get the uv indexes for j and i
            uv_j, uv_i, uv_j_c, uv_i_c = SetData(dl, K)
            # Perform RANSAC to remove outliers
            homography_inliers = homography_RANSAC(uv_j[:2,:].T, uv_i[:2,:].T)
            print("Total number of features: ",uv_j.shape[1])
            print("Number of inliers from Homography RANSAC: ",len(homography_inliers))

            inliers, num_inliers = GetInlierRANSAC(uv_j[:2,:].T,uv_i[:2,:].T,homography_inliers)
            print("Number of inliers from 8-pt RANSAC: ", num_inliers)
            getMatches(dl, inliers, n, match_idx, DATA_DIR)
            uv_j = uv_j.T[:, :2]
            uv_i = uv_i.T[:, :2]
            print("Shape uv_j", uv_j.shape)
            # store indexes of array which need triangulation
            needs_triangulation_idxs_list = []
            # for each row in uv_j
            X_i,x_i,master_list, needs_triangulation_idxs_list = checkNewFeatures(
                uv_i,
                uv_j,
                master_list,
                i,j,
                X_i,
                x_i,
                needs_triangulation_idxs_list
            )
            # print("Needs Triangulation len: ", len(needs_triangulation_idxs_list))
            # print("Needs triangulation list: ", needs_triangulation_idxs_list)
            triangulate_j_list.append(needs_triangulation_idxs_list)

        # Calculate the P matrix
        P_i, inlier_idxs, R_i, t_i = PnPRANSAC(X_i, x_i, K)
        print("R RANSAC: ",R_i)
        print("C RANSAC: ",t_i)
        print("No. of inliers: ", len(inlier_idxs))
        R_list.append(R_i)
        C_list.append(t_i)
        image_points.append(x_i)
        world_points.append(X_i)
        inliers_total.append(inlier_idxs)

        world_points_data = np.vstack(
            (X_i[inlier_idxs, :].T, np.ones((1, X_i[inlier_idxs, :].shape[0])))
        )
        ## initial Condition
        x_init = init_optimization_pose(C_list[-1], R_list[-1])

        # Optimization problem
        t_new, R_new = cameraCalibrationPose(
            x_i[inlier_idxs, :].T, K, x_init, world_points_data[0:3, :]
        )
        C_list.append(t_new)
        R_list.append(R_new)
        print("Non linear R :", R_new)
        print("Non linear C", t_new)
        # x_init = init_optimization_variables(x_trans_opt, R_quaternion_opt, X_i)

    print(-R_list[1].T @ C_list[1])
    print(-R_list[2].T @ C_list[2])
    #################################################
    #     ## Uncomment the below lines
    #     # # Triangulate to get new world points
    #     # for j in range(1,i):
    #     #     # obtain the index of the data list match images (j,i)
    #     #     match_idx = (j-1)*(10-j)/2 + i-j-1
    #     #     # get the list matching[ji]
    #     #     print("Match idx", match_idx)
    #     #     dl = data_list[int(match_idx)]
    #     #     # get the uv indexes for j and i
    #     #     uv_j, uv_i, uv_j_c, uv_i_c = SetData(dl,K)
    #     #     # Perform RANSAC to remove outliers (need to implement)

    #     #     # Perform triangulation to get new world points
    #     #     print(triangulate_j_list[j-1])
    #     #     AA = uv_j[:,triangulate_j_list[j-1]]
    #     #     print(type(AA))
    #     #     print(AA)
    #     #     # BB = uv_i[:,triangulate_j_list[j-1]]
    #     #     pts3D_4xN = triangulatePoints(uv_j[:,triangulate_j_list[j-1]], uv_i[:,triangulate_j_list[j-1]], P[j-1], P[i-1])
    #     #     # Perform non-linear triangulation (need to implement)

    #     #     # Store the world points, img ids and img pixels in master list (need to implement)

    # with open("./P2Data/Matches/master_list.txt", "w", newline="") as file:
    #     writer = csv.writer(file, delimiter=" ")
    #     # Write each list as a row
    #     writer.writerows(master_list)

    # R_total = np.array(R_total)
    # t_total = np.array(t_total)
    # # print(R_total[0, :, :])
    # print((t_total[0, :]))
    # print(R_total[0, :, :].T @ (-t_total[0, :]))
    # print(R_quaternion_opt.T @ (-x_trans_opt))
    # # print(R_ransac.T @ (-t_ransac))

    # world_points_3 = np.vstack(
    #     (world_points[0].T, np.ones((1, world_points[0].shape[0])))
    # )

    # # creating new points

    # data_list_12 = np.vstack((np.array(data_list[1]), np.array(data_list[4]))).tolist()

    # show_projection_image(
    #     t_total[0, :],
    #     R_total[0, :, :],
    #     world_points_3,
    #     K,
    #     DATA_DIR,
    #     data_list[1],
    #     n,
    #     img_n,
    #     inliers_total[0],
    #     "Linear 3",
    #     3,
    # )

    # show_projection_image(
    #     x_trans_opt,
    #     R_quaternion_opt,
    #     pts3D_4xN_casadi,
    #     K,
    #     DATA_DIR,
    #     data_list[0],
    #     n,
    #     img_n,
    #     inliers_index,
    #     "Nolinear 2",
    #     2,
    # )


if __name__ == "__main__":
    main()
