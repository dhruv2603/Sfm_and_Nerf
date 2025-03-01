import numpy as np
from LinearTriangulation import LinearTriangulation
def DisambiguateCameraPose(R_l, C_l, X_list):
    """
    Extract the correct camera pose from the combination of 4 camera pose pairs
    Inputs: R_l    - List of rotation matrices
            C_l    - List of Camera translation vectors
            X_list - List of world points for each set of R,C pairs
    Output: C      - correct camera translation vector
            R      - correct camera rotation matrix
            X      - correct set of world points
    """
    inliers_max = 0
    for r,c,x in zip(R_l,C_l,X_list):
        # print(np.array(x).shape)
        # print(type(x))
        condition1 = np.array(x)[:,2]>0
        condition2 = np.array(r)[:,2].T @ (np.array(x).T - np.array(c).reshape(-1,1)>0)
        inliers = np.logical_and(condition1,condition2)

        if np.sum(inliers) > np.sum(inliers_max):
            inliers_max = inliers
            R = r
            C = c 
            X = x   
    return C,R,X