import numpy as np
import random
from LinearPnP import *
from IPython import embed

def PnPRANSAC(x,X,K):
    # x: 2D image feature coordinate
    # X: 3D coordinate
    num_iters = 10
    epsilon = 15
    n = 0
    for i in range(num_iters):
        indices = random.sample(range(X.shape[0]),6)
        x_sample = x[indices]
        X_sample = X[indices]
        R,C = LinearPnP(x_sample,X_sample,K)
        inliers = []
        T = -R@C.reshape(3,1)
        P = np.hstack((R,T))
        P = K@P
        P = np.vstack((P,[0,0,0,1]))
        for j in range(X.shape[0]):
            # print(X.shape)
            # print(X[j].shape)
            # embed()
            X_3d = (np.array(X[j])).reshape(3,1)
            X_3d = np.vstack((X_3d,1))
            u = x[j][0]
            v = x[j][1]
            img_coor = P@X_3d
            img_coor = img_coor/img_coor[-1]
            u_new = img_coor[0]
            v_new = img_coor[1]
            error = (u-u_new)**2 + (v-v_new)**2
            if error<epsilon:
                inliers.append(j)
        if len(inliers)>n:
            n = len(inliers)
            final_inliers = inliers
    
    R_new,C_new = LinearPnP(x[final_inliers],X[final_inliers],K)
    return R_new,C_new    










    
