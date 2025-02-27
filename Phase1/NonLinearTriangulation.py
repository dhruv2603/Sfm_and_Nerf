import numpy as np
import scipy
import scipy.optimize

def NonLinearTriangulation(dl, K, R, C, X0):
    """
    This function calculates world coordinates using the Non-LinearTriangulation formulation by minimizing the reprojection error.
    Inputs: dl - camera image points
            K  - Intrinsic Matrix of the camera
            R  - Rotation of camera 2
            C  - Translation of camera 2
            X0 - Initial guess of world coordinates obtained using Linear Triangulation
    Output: X  - Final world coordinates
    """
    def ResidualFunction(x0,data,P1,P2):
        e_1 = 0
        e_2 = 0
        u_1 = float(data[0])
        v_1 = float(data[1])
        u_2 = float(data[2])
        v_2 = float(data[3])
        x0 = np.reshape(x0,(-1,1))
        e_1 = np.array([(u_1 - (np.matmul(P1[0,:],x0)/np.matmul(P1[2,:],x0)))**2,(v_1 - (np.matmul(P1[1,:],x0)/np.matmul(P1[2,:],x0)))**2])
        e_2 = np.array([(u_2 - (np.matmul(P2[0,:],x0)/np.matmul(P2[2,:],x0)))**2,(v_2 - (np.matmul(P2[1,:],x0)/np.matmul(P2[2,:],x0)))**2])
        test = (u_1 - (np.matmul(P1[0,:],x0)/np.matmul(P1[2,:],x0)))**2
        e = np.vstack([e_1,e_2]).flatten()
        return e

    R0 = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,1]])
    C0 = np.array([[0],
                   [0],
                   [0]])
    P1 = np.matmul(K,np.hstack((R0,-np.matmul(R0,C0))))
    P2 = np.matmul(K,np.hstack((R,-np.matmul(R,C))))
    X = []
    for i,data in enumerate(dl):
        x0 = np.hstack((X0[i],1))
        x0 = x0.flatten()
        result = scipy.optimize.least_squares(ResidualFunction,x0 = x0, method = "lm", args = (data,P1,P2))
        world_coords = result.x
        world_coords = world_coords/world_coords[3]
        X.append(world_coords[:3])
    return X