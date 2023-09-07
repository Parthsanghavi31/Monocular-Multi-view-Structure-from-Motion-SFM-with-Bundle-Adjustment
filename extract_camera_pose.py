from IPython import embed
import argparse
import random
import Utils.FeatureUtils as FeatureUtils
import numpy as np

def extract_cam_pose(E):
    Cs = [] 
    Rs = []
    W = np.array([[0 ,-1, 0],[1,0,0],[0,0,1]])
    U,S,V = np.linalg.svd(E,full_matrices=True)
    # print(U,V)

    # append centers
    Cs.append(U[:][2])
    Cs.append(-U[:][2])
    Cs.append(U[:][2])
    Cs.append(-U[:][2])

    # append rotations
    Rs.append(U@W@V)
    Rs.append(U@W@V)
    Rs.append(U@W.T@V)
    Rs.append(U@W.T@V)

    # print(Cs,Rs)
    Rn,Cn = chk_det(Rs,Cs)
    
    return Rn,Cn

def chk_det(Rs,Cs):
    Rn,Cn = [],[]
    assert(len(Rs) == len(Cs))
    print("Number of R and T: ", len(Rs))
    for R,C in zip(Rs,Cs):
        if(np.round(np.linalg.det(R))==-1):
            Rn.append(-R)
            Cn.append(-C)
        else:
            Rn.append(R)
            Cn.append(C)
    return Rn,Cn

    