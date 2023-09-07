import numpy as np
from IPython import embed


def LinearPnP(image_points,world_points,K,i=0):
    # X: 3D point
    # x: 2D point in Image plane
    world_points = np.array(world_points)
    image_points = np.array(image_points)

    X = np.array(world_points[:,0])
    Y = np.array(world_points[:,1])
    Z = np.array(world_points[:,2])
    
    u = np.array(image_points[:,0])
    v = np.array(image_points[:,1])

    # print(world_points.shape,image_points.shape)
    zero = np.zeros_like(X)
    one = np.ones_like(X)
    # print(X.shape,u.shape,zero.shape)
    A1 = np.array([X,Y,Z,one,zero,zero,zero,zero,-X*u,-Y*u,-Z*u,-u]).T
    A2 = np.array([zero,zero,zero,zero,X,Y,Z,one,-X*v,-Y*v,-Z*v,-v]).T
    
    A = np.vstack((A1,A2))
    U,S,V = np.linalg.svd(A)
    P = V[np.argmin(S),:]

    P = P.reshape((3,4))
    R = np.linalg.inv(K)@P[0:3,0:3]
    UR,SR,VR = np.linalg.svd(R)

    R_new = UR@VR
    T = np.linalg.inv(K)@P[:,3]
    
    Rdet = np.linalg.det(R_new)
    
    if Rdet<0:
        R_new = -R_new
        T = -T
    
    C = -R_new@T
    
    return R_new, T 


