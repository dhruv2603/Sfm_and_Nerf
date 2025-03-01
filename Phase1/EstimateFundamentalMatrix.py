import numpy as np
import casadi as ca
from scipy.spatial.transform import Rotation as R
import cv2 as cv2
from numpy import linalg as la


def getFundamentalMatrix(ptsA, ptsB, num_point=8):

    # Convert data type of points to float64
    ptsA = np.float64(ptsA)
    ptsB = np.float64(ptsB)

    # Sample num_point points
    sample_ptsA, sample_ptsB = randomSampleCorrPoint(ptsA, ptsB, num_point)

    # Get normalise matrix based on the sample points
    normalisationMat_A = getNormalisationMat(ptsA)
    normalisationMat_B = getNormalisationMat(ptsB)

    # Convert points to homogeneous coordinates
    sample_ptsA, sample_ptsB = conv2HomogeneousCor(sample_ptsA, sample_ptsB)

    # Normalise data points
    sample_ptsA_normal = np.float64(
        [normalisationMat_A @ s_ptA for s_ptA in sample_ptsA]
    )
    sample_ptsB_normal = np.float64(
        [normalisationMat_B @ s_ptB for s_ptB in sample_ptsB]
    )

    # Compute the design matrix
    design_matrix = np.array(
        [
            (np.expand_dims(b, axis=1) @ np.expand_dims(a, axis=0)).flatten()
            for a, b in zip(sample_ptsA_normal, sample_ptsB_normal)
        ]
    )

    # SVD the design matrix
    U_des, s_des, VT_des = la.svd(design_matrix)

    # Find the vector V_i with the least cooresponding singular value
    f_vec = VT_des[-1, :]

    # Reform draft F from V_i
    f = np.float64(f_vec.reshape((3, 3)))

    # SVD draft F
    U_f, s_f, VT_f = la.svd(f)

    # Set the least singular value of draft F to 0
    s_f[-1] = 0
    s_f_new = np.diag(s_f)

    # re-Construct F using the new S and draft U, draft V
    F_n = U_f @ s_f_new @ VT_f

    # De-normalise
    F = normalisationMat_B.T @ F_n @ normalisationMat_A
    F = F / F[-1, -1]

    return F


def getNormalisationMat(pts):

    pts = np.float64(pts)
    mean = np.array(np.sum(pts, axis=0) / len(pts), dtype=np.float64)
    scale = np.sum(la.norm(pts - mean, axis=1), axis=0) / (len(pts) * np.sqrt(2.0))
    normalisationMat = np.array(
        [
            [1.0 / scale, 0.0, -mean[0] / scale],
            [0.0, 1.0 / scale, -mean[1] / scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return normalisationMat


def randomSampleCorrPoint(ptsA, ptsB, num_point=8):

    if num_point >= len(ptsA):
        return ptsA, ptsB
    else:
        rng = np.random.default_rng()
        ponit_index = rng.choice(np.arange(len(ptsA)), size=num_point, replace=False)
        sample_ptsA = ptsA[ponit_index, :]
        sample_ptsB = ptsB[ponit_index, :]
        return sample_ptsA, sample_ptsB


def conv2HomogeneousCor(ptsA, ptsB):

    if ptsA.ndim == 1:
        ptsA_homo = np.pad(ptsA, (0, 1), "constant", constant_values=1.0)
        ptsB_homo = np.pad(ptsB, (0, 1), "constant", constant_values=1.0)
    else:
        ptsA_homo = np.pad(ptsA, [(0, 0), (0, 1)], "constant", constant_values=1.0)
        ptsB_homo = np.pad(ptsB, [(0, 0), (0, 1)], "constant", constant_values=1.0)

    return np.float64(ptsA_homo), np.float64(ptsB_homo)


def normalizationMatrix(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts)
    Norm_Mat = np.array(
        [[1 / std, 0, -mean[0] / std], [0, 1 / std, -mean[1] / std], [0, 0, 1]]
    )
    normalized_points = np.hstack((pts, np.ones((pts.shape[0], 1))))
    normalized_points = (Norm_Mat @ normalized_points.T).T
    return normalized_points, Norm_Mat


def EstimateFundamentalMatrix(pixels_1, pixels_2):
    """
    Estimate the funcdamental matrix
    Inputs: pixels_1 - (8,2) matrix of pixel values of the features in image 1
            pixels_2 - (8,2) matrix of pixel values of the features in image 2
    Output: F        - The fundamental matrix
    """

    norm_pixels_1, M1 = normalizationMatrix(pixels_1)
    norm_pixels_2, M2 = normalizationMatrix(pixels_2)

    x_i, y_i = norm_pixels_1[:, 0], norm_pixels_1[:, 1]

    x_j, y_j = norm_pixels_2[:, 0], norm_pixels_2[:, 1]

    ones = np.ones(x_i.shape[0])
    A = [x_i * x_j, y_i * x_j, x_j, x_i * y_j, y_i * y_j, y_j, x_i, y_i, ones]
    A = np.vstack(A).T
    U, sigma, V = np.linalg.svd(A)
    f = V[np.argmin(sigma), :]
    f = f.reshape((3, 3))
    Uf, Df, Vf = np.linalg.svd(f)
    Df[-1] = 0
    F = Uf @ np.diag(Df) @ Vf

    F = M2.T @ F @ M1
    F = F / F[2, 2]

    return F
