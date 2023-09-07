import numpy as np
from IPython import embed
from scipy.optimize import least_squares



def non_linear_triangulation(inlier_matches, X_world, R1, C1, R2, C2, K):
    
    T1 = -1 * np.dot(R1, C1)
    P1 = np.dot(K, np.hstack((R1, T1)))

    C2 = np.expand_dims(C2, 1)
    T2 = -1 * np.dot(R2, C2)
    P2 = np.dot(K, np.hstack((R2, T2)))

    count = 0
    X3D_new = []
    for i in range(len(X_world)):
        result_x = least_squares(calculate_reprojection_error, x0=X_world[i], args=[inlier_matches[i], P1, P2])
        # loss='soft_l1',
        # if result_x.success:
        #     print(result_x.fun)
        #     count += 1
        # else:
        #     print("False: ", result_x.fun)

        X3D_new.append(result_x.x)
    
    return np.array(X3D_new)


def calculate_reprojection_error(X, feature_pair, P1, P2):
    
    error1 = calculate_image_error(feature_pair[0], X, P1)
    error2 = calculate_image_error(feature_pair[1], X, P2)
    total_error = error1 + error2

    return total_error


def calculate_image_error(uv, X, P):
    uv = np.concatenate((uv, np.array([1])))
    uv = np.expand_dims(uv, 1)
    X = np.concatenate((X, np.array([1])))
    X = np.expand_dims(X, 1)

    uv_tilda = P @ X
    uv_tilda = uv_tilda / uv_tilda[2]
    diff = uv - uv_tilda
    error = diff[0]**2 + diff[1]**2
    return error

