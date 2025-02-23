import numpy as np
import os
import scipy
import scipy.linalg

def EstimateFundamentalMatrix(dl):
    """
    This function calculates the fundamental matrix that represents the epipolar geometry between the camera poses.
    Input : List of u,v image pixel values for the two image views
    Output: The fundamental matrix F
    """
    A = np.empty([0,9])
    for pt in dl:
        u = float(pt[0])
        v = float(pt[1])
        u_prime = float(pt[2])
        v_prime = float(pt[3])
        row = np.array([u*u_prime, u*v_prime, u, v*u_prime, v*v_prime, v, u_prime, v_prime, 1])
        A = np.vstack((A,row))
    _,_,V = scipy.linalg.svd(A)
    F = V[-1].reshape(3,3)
    Uf,Df,Vf = scipy.linalg.svd(F)
    Df[-1] = 0
    F = np.matmul(Uf,np.matmul(np.diag(Df),Vf))
    return F