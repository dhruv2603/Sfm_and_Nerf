import numpy as np
import os
import random
from tqdm import tqdm
from EstimateFundamentalMatrix import getFundamentalMatrix, EstimateFundamentalMatrix


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


def getCorrespondencesEpilines(ptsA, ptsB, FundMat):

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

def GetInlierRANSAC(pixels1,pixels2,homo_inliers,N=2000,tau=5):
    """
    Get the inliers using RANSAC with the Fundamental Matrix equation.
    Inputs: pixels1 - (M,2) array of pixel values of image 1
            pixels2 - (M,2) array of pixel values of image 2
            inliers - the list of inliers obtained from homography RANSAC
            N       - Number of iterations
            tau     - Threshold
    """
    print("Running RANSAC Iterations for Feature Inliers")
    for i in tqdm(range(N)):
        num_inliers = 0
        curr_max = 0
        inliers_i = []
        inliers = []
        idx = np.random.randint(0,pixels1.shape[0],8)
        rand_eight_pts_1 = pixels1[idx, :]
        rand_eight_pts_2 = pixels2[idx, :]
        
        F = EstimateFundamentalMatrix(rand_eight_pts_1, rand_eight_pts_2)
        
        for j, (pt1, pt2) in enumerate(zip(pixels1, pixels2)):
            x1, y1 = pt1[0], pt1[1]
            x2, y2 = pt2[0], pt2[1]
            
            X1 = np.array([x1, y1, 1])
            X2 = np.array([x2, y2, 1])
            
            val = X1.T @F @ X2 #X2 @ F @ X1
            
            if(abs(val) < tau and j in homo_inliers):
                num_inliers +=1
                inliers_i.append(j)
                
        if (num_inliers > curr_max):
            curr_max = num_inliers
            inliers = inliers_i
    
    return inliers, curr_max