import numpy as np
import scipy
import cv2 as cv2
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from LinearTriangulation import triangulatePoints

def ExtractCameraPose(E):
    """
    Extract camera poses from Essential Matrix
    Input:  E            - Essential Matrix
    Output: camera_poses - possible combinations of R and C
    """
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

    camera_poses = []
    for i, item in enumerate(R_candidates):
        if np.linalg.det(item) > 0:
            camera_poses.append((t_candidates[0], R_candidates[i]))
            camera_poses.append((t_candidates[1], R_candidates[i]))
            
        else:
            camera_poses.append((-t_candidates[0], -R_candidates[i]))
            camera_poses.append((-t_candidates[1], -R_candidates[i]))

    return camera_poses

# def recoverPoseFromFundamental(E, K, pts1, pts2):
#     # Convert input points to homogeneous coordinates for triangulation.
#     N = pts1.shape[0]
#     pts1_h = np.vstack((pts1.T, np.ones((1, N))))  # Shape: (3, N)
#     pts2_h = np.vstack((pts2.T, np.ones((1, N))))  # Shape: (3, N)
    
#     F_identity = np.eye(3)
#     Identity = np.hstack([np.eye(3), np.zeros((3, 1))])  # 3x4 matri
#     I = np.eye(3, 3)
#     t = np.zeros((3, 1))
#     aux_last_element_homogeneous = np.array([[0.0, 0.0, 0.0, 1.0]])
#     T_1 = np.vstack(
#         (
#             np.hstack((I, t)),  # shape: (3,4)
#             aux_last_element_homogeneous,
#         )
#     )

#     P1 = K @ F_identity @ Identity @ T_1

#     bestCount = -np.inf
#     best_R = None
#     best_t = None
#     best_inliers = None

#             T_2 = np.vstack(
#                 (
#                     np.hstack((R_test, t_test.reshape((3, 1)))),  # shape: (3,4)
#                     aux_last_element_homogeneous,
#                 )
#             )
#             # Camera 2 projection matrix: [R | t]
#             P2 = K @ F_identity @ Identity @ T_2

#             # Triangulate points.
#             # pts3D = cv2.triangulatePoints(
#             #    P1[0:3, 0:4],
#             #    P2[0:3, 0:4],
#             #    pts1_h[0:2, :],
#             #    pts2_h[0:2, :],
#             # )  # OpenCV's Linear-Eigen triangl
#             # pts3D = pts3D / pts3D[3, :]

#             pts3D = triangulatePoints(
#                 pts1_h[0:3, :],
#                 pts2_h[0:3, :],
#                 P1[0:3, 0:4],
#                 P2[0:3, 0:4],
#             )  # OpenCV's Linear-Eigen triangl
#             pts3D = pts3D / pts3D[3, :]

#             # Check depth in camera 1 (Z1 > 0).
#             Z1 = pts3D[2, :]
#             # Check depth in camera 2 (Z2 > 0): transform pts3D into camera 2 frame.
#             pts3D_cam2 = R_test @ pts3D[0:3, :] + t_test.reshape(3, 1)
#             Z2 = pts3D_cam2[2, :]

#             valid = (Z1 > 0) & (Z2 > 0)
#             numInFront = np.sum(valid)

#             if numInFront > bestCount:
#                 bestCount = numInFront
#                 best_R = R_test
#                 best_t = t_test
#                 best_inliers = valid

#     R = best_R
#     t = best_t
#     inliers = best_inliers
#     return R, t, inliers
