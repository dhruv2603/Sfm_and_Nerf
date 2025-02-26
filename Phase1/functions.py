import os
import numpy as np
from numpy import linalg as la
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt


def randomSampleCorrPoint(ptsA, ptsB, num_point=8):
    """Radomly sample corespondences from the given data set

    Parameters
    ----------
    ptsL : int numpy.ndarray, shape (n_correspondences, 2)
        An array of coordinates of correspondences from the left image.
    ptsR : int numpy.ndarray, shape (n_correspondences, 2)
        An array of coordinates of correspondences from the right image.
    num_point: int
        A number specifices the amount of ponts to sample from each array ptsL and ptsR. default = 8
    -------
    Return
    sample_ptsL : int numpy.ndarray, shape (num_point, 2)
        An array of coordinates of sample correspondences from the left image.  If num_point is greater than
        ptsL length, the original point set is return
    sample_ptsR : int numpy.ndarray, shape (num_point, 2)
        An array of coordinates of sample correspondences from the right image. If num_point is greater than
        ptsL length, the original point set is return
    """

    if num_point >= len(ptsA):
        return ptsA, ptsB
    else:
        rng = np.random.default_rng()
        ponit_index = rng.choice(np.arange(len(ptsA)), size=num_point, replace=False)
        sample_ptsA = ptsA[ponit_index, :]
        sample_ptsB = ptsB[ponit_index, :]
        return sample_ptsA, sample_ptsB


def conv2HomogeneousCor(ptsA, ptsB):
    """Convert points from cartesian coordinate to homogeneous coordinate

    Parameters
    ----------
    ptsA : int numpy.ndarray, shape (n_correspondences, 2) or int numpy.ndarray, shape (2,)
        A coordinate or an array of coordinates of correspondences from image A.
    ptsB : int numpy.ndarray, shape (n_correspondences, 2) or int numpy.ndarray, shape (2,)
        A coordinate or an array of coordinates of correspondences from image B.
    -------
    Return
    ptsA_homo : float64 numpy.ndarray, shape (n_correspondences, 3) or int numpy.ndarray, shape (3,)
        A coordinate or an array of coordinates of correspondences from image A, in the form of homogeneous coordinate.
    ptsB_homo : float64 numpy.ndarray, shape (n_correspondences, 3) or int numpy.ndarray, shape (3,)
        A coordinate or an array of coordinates of correspondences from image B, in the form of homogeneous coordinate.
    """

    if ptsA.ndim == 1:
        ptsA_homo = np.pad(ptsA, (0, 1), "constant", constant_values=1.0)
        ptsB_homo = np.pad(ptsB, (0, 1), "constant", constant_values=1.0)
    else:
        ptsA_homo = np.pad(ptsA, [(0, 0), (0, 1)], "constant", constant_values=1.0)
        ptsB_homo = np.pad(ptsB, [(0, 0), (0, 1)], "constant", constant_values=1.0)

    return np.float64(ptsA_homo), np.float64(ptsB_homo)


def getNormalisationMat(pts):
    """Calculate the nomalisation matrix of the given coordinate points set

    Parameters
    ----------
    pts : int numpy.ndarray, shape (n_correspondences, 2)
        An array of coordinate points.
    -------
    Return
    normalisationMat : float numpy.ndarray, shape (3, 3)
        The normalisation matrix of the given point set.
        This matrix translate and scale the points so that the mean coordinate is at (0,0) and average distance to (0,0) is sqrt(2)
    """

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


def getFundamentalMatrix(ptsA, ptsB, num_point=8):
    """Radomly sample corespondences from the given data set

    Parameters
    ----------
    ptsA : int numpy.ndarray, shape (n_correspondences, 2)
        An array of coordinates of correspondences from image A.
    ptsB : int numpy.ndarray, shape (n_correspondences, 2)
        An array of coordinates of sample correspondences from the image B.
    num_point : int, default = 8
        The number of correspondences from the entire point set being used to calculate the fundamental matrix
    -------
    Return
    F : float numpy.ndarray, shape (3, 3)
        Fundamental matrix based on the sample correspondences.
    """

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


def getCorrespondencesEpilines(ptsA, ptsB, FundMat):
    """Compute the epipolar lines on image A and B based on the
       correspondences and the fundamental matrix

    Parameters
    ----------
    ptsA : int numpy.ndarray, shape (n_correspondences, 3) or int numpy.ndarray, shape (3,)
        A coordinate or an array of coordinates of correspondences from image A.
    ptsB : int numpy.ndarray, shape (n_correspondences, 3) or int numpy.ndarray, shape (3,)
        A coordinate or an array of coordinates of correspondences from image B.
    F : float numpy.ndarray, shape (3, 3)
        Fundamental matrix.
    -------
    Return
    lines : float, numpy.ndarray, shape (num_points, 3)
        An array of the epipolar lines.  Each epipolar line is represented as an array of
        three float number [a, b, c].  [a, b, c] are the coefficients of a line ax + by + c = 0.
        Lines are normalised ao that sqrt(a^2 + b^2) = 1
    """

    # Convert data type to float64
    ptsA = np.float64(ptsA)
    ptsB = np.float64(ptsB)

    # If input is only a point
    if ptsA.ndim == 1:
        # Compute the lines
        linesA = np.array(ptsB @ FundMat, dtype=np.float64)
        linesB = np.array(FundMat @ ptsA.T, dtype=np.float64)

        # Normalise
        aA, bA, cA = linesA
        aB, bB, cB = linesB
        linesA = linesA / np.sqrt(aA * aA + bA * bA)
        linesB = linesB / np.sqrt(aB * aB + bB * bB)
    else:
        # Compute the lines
        linesA = np.array([pB @ FundMat for pB in ptsB], dtype=np.float64)
        linesB = np.array([FundMat @ pA.T for pA in ptsA], dtype=np.float64)

        # Normalise
        linesA = np.array(
            [
                np.array(
                    [
                        a / np.sqrt(a * a + b * b),
                        b / np.sqrt(a * a + b * b),
                        c / np.sqrt(a * a + b * b),
                    ],
                    dtype=np.float64,
                )
                for a, b, c in linesA
            ],
            dtype=np.float64,
        )
        linesB = np.array(
            [
                np.array(
                    [
                        a / np.sqrt(a * a + b * b),
                        b / np.sqrt(a * a + b * b),
                        c / np.sqrt(a * a + b * b),
                    ],
                    dtype=np.float64,
                )
                for a, b, c in linesB
            ],
            dtype=np.float64,
        )

    return linesA, linesB


def getFundamentalMatRANSAC(ptsA, ptsB, tol, num_sample=8, confidence=0.99):
    """Calculate the best fundamental Matrix for given correspondences using RANSAC

    Parameters
    ----------
    ptsA : int numpy.ndarray, shape (n_correspondences, 2)
        An array of coordinates of correspondences from image A.
    ptsB : int numpy.ndarray, shape (n_correspondences, 2)
        An array of coordinates of correspondences from image B.
    tol : float
        the tolerance distance that allows a point to deviate from the epipolar line
        and still be considered as inlier.  Inliers have distance <= tol from the epipolar line.
    num_point : int, default = 8
        The number of correspondences from the entire point set being used to calculate the fundamental matrix
    confidence : float, default = 0.99, 0 < confidence < 1
        The confidence of getting the fundamental matrix from subset of points that are all inliers
    -------
    Return
    best_F : float numpy.ndarray, shape (3, 3)
        The best fundamental matrix of the given correspondeces.
    best_inlier: int {0, 1} numpy.ndarray, shape (length_ptsL,)
        An array of 0 and 1, 1 means the coorespondence at the same index are inliers,
        0 means outliers.
    """

    best_inlier_num = 0
    best_inlier = np.zeros(len(ptsA))
    tol = np.float64(tol)

    # Iteration is calculated based on the confidence and the asumption that 50% correspondences are inliers
    # and 50% correspondences are outliers.
    iterations = int(
        np.ceil(
            np.log10(1 - confidence) / np.log10(1 - np.float_power(0.5, num_sample))
        )
    )

    for _ in tqdm(range(iterations)):
        sample_ptsA, sample_ptsB = randomSampleCorrPoint(ptsA, ptsB, num_sample)

        F = getFundamentalMatrix(sample_ptsA, sample_ptsB)

        inlier = np.zeros(len(ptsA), dtype=np.float64)

        for i, (ptA, ptB) in enumerate(zip(ptsA, ptsB)):
            # Convert to homogeneous coordinate
            ptA_homo, ptB_homo = conv2HomogeneousCor(ptA, ptB)

            # Get the epipolar lines
            l_A, l_B = getCorrespondencesEpilines(ptA_homo, ptB_homo, F)
            l_A = np.float64(l_A)
            l_B = np.float64(l_B)

            # Calculate the error
            err_A = np.float64(abs(l_A @ ptA_homo))
            err_B = np.float64(abs(l_B @ ptB_homo))
            if err_A <= tol and err_B <= tol:
                inlier[i] = 1

        if np.sum(inlier) > best_inlier_num:
            best_inlier = inlier
            best_inlier_num = np.sum(inlier)
            best_F = F

    return best_F, best_inlier
