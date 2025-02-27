import numpy as np
from LinearTriangulation import LinearTriangulation
def DisambiguateCameraPose(dl, K, R_l, C_l):
    c_max = 0
    world_poses = []
    R = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])
    C = np.array([[0],
                  [0],
                  [0]])
    for r,c in zip(R_l,C_l):
        c = np.reshape(c,(3,1))
        X_list, count = LinearTriangulation(dl,K,r,c)
        if count > c_max:
            c_max = count
            world_coords = X_list
            R = r
            C = c
    return R,C,world_coords