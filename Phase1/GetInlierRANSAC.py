import numpy as np
import os
import random
import cv2
from tqdm import tqdm
from EstimateFundamentalMatrix import EstimateFundamentalMatrix

def GetInlierRANSAC(pts1,pts2,Tau=0.01, N=100):
    """
    Get Inlier points using RANSAC
    Input : pts1,pts2, Tau, M
    Output: Inlier points
    """
    n  = 0
    sz = pts1.shape[0]
    print("Feature Size is: ",sz)
    inliers      = []
    inliers_idxs = []
    for _ in tqdm(range(N)):
        # indices = random.sample(range(pts1.shape[0]), 8)
        #Get 8 random points
        indices = np.random.choice(pts1.shape[0], 8, replace=False)
        # Use the indices to select rows from both arrays
        pts1_rand = pts1[indices]
        pts2_rand = pts2[indices]
        F = EstimateFundamentalMatrix(pts1_rand,pts2_rand)
        set_in      = []
        set_in_idxs = []
        for i in range(sz):
            u_i  = np.array([pts1[i,0], pts1[i,1], 1]).reshape(3,1)
            v_i  = np.array([pts2[i,0], pts2[i,1], 1]).reshape(3,1)
            #Cost = np.matmul(np.matmul(v_i,F),np.transpose(u_i))
            Cost = np.abs(u_i.T @ F @ v_i)
            # if abs(Cost.item()) <= Tau:
            if Cost.item() <= Tau:
                set_in_idxs.append(i)
            if len(set_in_idxs) >= 0.9*sz:
                Fin = F
                print("Found fundamental Matrix with inliers 0.9 of total features")
                return Fin, set_in_idxs
        if len(set_in_idxs) > n:
            n            = len(set_in_idxs)
            inliers_idxs = set_in_idxs
            Fin = F
    print("Maximum inliers found is: ",len(inliers_idxs))
    return Fin, inliers_idxs