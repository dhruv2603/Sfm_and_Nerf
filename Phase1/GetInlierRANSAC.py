import numpy as np
import os
import random
from tqdm import tqdm
from EstimateFundamentalMatrix import EstimateFundamentalMatrix

def GetInlierRANSAC(dl,Tau=0.5*1e-3, N=1000):
    """
    Get Inlier points using RANSAC
    Input : dl, Tau, M
    Output: Inlier points
    """
    n  = 0
    sz = len(dl)
    print("Feature Size is: ",sz)
    inliers      = []
    inliers_idxs = []
    for _ in tqdm(range(N)):
        random_pixels = random.sample(dl,8)
        F = EstimateFundamentalMatrix(random_pixels)
        set_in      = []
        set_in_idxs = []
        for i in range(sz):
            u_i  = np.array([float(dl[i][0]), float(dl[i][1]), 1]).reshape(3,1)
            v_i  = np.array([float(dl[i][2]), float(dl[i][3]), 1]).reshape(3,1)
            Cost = np.matmul(np.matmul(np.transpose(u_i),F),v_i)
            if abs(Cost.item()) <= Tau:
                set_in.append(dl[i])
                set_in_idxs.append(i)
            if len(set_in) >= 0.9*sz:
                print("Found fundamental Matrix with inliers 0.9 of total features")
                print(len(set_in))
                return set_in, set_in_idxs
        if len(set_in) > n:
            n            = len(set_in)
            inliers      = set_in
            inliers_idxs = set_in_idxs
    print("Maximum inliers found is: ",len(inliers))
    return inliers, inliers_idxs