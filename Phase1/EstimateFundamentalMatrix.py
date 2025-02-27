import numpy as np
import os
import scipy
import scipy.linalg
import cv2

def Norm_pts(pts):
    mean_pts = np.mean(pts,axis=0)
    mean_0_pts = pts - mean_pts
    dist = np.mean(np.linalg.norm(mean_0_pts, axis=1))
    scale = np.sqrt(2)/dist
    scaled_pts = mean_0_pts*scale
    homo_pts = np.hstack([scaled_pts,np.ones((scaled_pts.shape[0],1))])
    T = np.array([[scale, 0, -scale*mean_pts[0]],
                  [0, scale, -scale*mean_pts[1]],
                  [0, 0, 1]])
    
    return homo_pts, T

def EstimateFundamentalMatrix(pts1,pts2):
    """
    This function calculates the fundamental matrix that represents the epipolar geometry between the camera poses.
    Input : List of u,v image pixel values for the two image views
    Output: The fundamental matrix F
    """
    norm_pts1,T1 = Norm_pts(pts1)
    norm_pts2,T2 = Norm_pts(pts2)
    A = np.empty([0,9])
    for i,pt in enumerate(norm_pts1):
        u = pt[0]
        v = pt[1]
        u_prime = norm_pts2[i][0]
        v_prime = norm_pts2[i][1]
        row = np.array([u*u_prime, v*u_prime, u_prime, u*v_prime, v*v_prime, v_prime, u, v, 1])
        A = np.vstack((A,row))
    _,_,V = scipy.linalg.svd(A)
    F = V[-1].reshape(3,3)
    Uf,Df,Vf = scipy.linalg.svd(F)
    Df[-1] = 0
    F = np.matmul(Uf,np.matmul(np.diag(Df),Vf))
    F = np.matmul(np.transpose(T2),np.matmul(F,T1))
    F = F/F[2,2]
    return F