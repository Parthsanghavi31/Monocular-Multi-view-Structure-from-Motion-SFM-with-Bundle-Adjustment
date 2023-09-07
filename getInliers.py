from IPython import embed
import argparse
import random
import Utils.FeatureUtils as FeatureUtils
import Utils.DataUtils as DataUtils
import numpy as np
import cv2

class Fundamental_Matrix:
    def __init__(self, args, matches):
        self.epsilon = 0.7                # epsilon
        self.num_iters = 1000             # M
        # self.num_points_ransac = 300    # N
        self.num_inliers = 0
        self.iter_inliers = {}
        self.final_inliers = {}
        self.F = np.zeros((3, 3))
        self.E = np.zeros((3, 3))
        
        self.data_utils = DataUtils(args.basePath)
        random.seed(40)

        # self.feature_utils = FeatureUtils()
        # self.matched_features = self.feature_utils.read_matching_files(args.basePath)
        self.matched_features = matches

    def estimate_fundamental(self, eight_points):
        A = []
        for point in eight_points:
            a,b = point[1]
            c,d = point[0]
            row = [a*c,c*b,c,d*a,d*b,d,a,b,1]
            A.append(row)
        A = np.array(A)

        u, s, vh = np.linalg.svd(A, full_matrices=True)

        F = vh[:][8]
        F = F.reshape((3,3))

        uf,s_f,vf = np.linalg.svd(F,full_matrices=True)
        s_f[-1] = 0
        s_f = np.diag(s_f)
        F_new = np.dot(uf,np.dot(s_f,vf))
        # Ft = uf @ s_f @ vf
        return F_new

    def perform_ransac(self, image_pair):
        count_inliers = 0
        self.num_inliers = 0
        for i in range(self.num_iters):
            self.iter_inliers.clear()

            point_pairs_8 = random.sample(self.matched_features[image_pair], 8)
            F = self.estimate_fundamental(point_pairs_8)

            point_pairs = self.matched_features[image_pair]
            # random.sample(self.matched_features[image_pair], self.num_points_ransac)
            
            for point_pair in point_pairs:
                point1 = point_pair[0]
                point2 = point_pair[1]
                point1 = np.expand_dims(np.array([point1[0], point1[1], 1]), axis=1)
                point2 = np.expand_dims(np.array([point2[0], point2[1], 1]), axis=1)

                product = abs(np.matmul(np.matmul(point2.T, F), point1))

                if product < self.epsilon:
                    count_inliers += 1
                    self.update_inliers(image_pair, point_pair)
                    # self.feature_utils.add_inliers(image_pair, point_pair)

            if (self.num_inliers < count_inliers):
                self.num_inliers = count_inliers
                self.final_inliers[image_pair] = self.iter_inliers[image_pair]
                self.F = F


        return self.final_inliers
    
    def update_inliers(self, img_idxs, point_pair):
        image_i_idx, image_j_idx = img_idxs
        image_i_u, image_i_v = point_pair[0]
        image_j_u, image_j_v = point_pair[1]
        if self.iter_inliers.get((image_i_idx, image_j_idx)) is not None:
            self.iter_inliers[(image_i_idx, image_j_idx)].append([(image_i_u, image_i_v), (image_j_u, image_j_v)])
        else:
            self.iter_inliers[(image_i_idx, image_j_idx)] = [[(image_i_u, image_i_v), (image_j_u, image_j_v)]]

    def get_F(self):
        return self.F
    
    def get_essential_from_fundamental(self):
        F = self.F
        K = self.data_utils.load_camera_instrinsics()
        E = np.matmul(np.matmul(K.T, F), K)
        U, S, V = np.linalg.svd(E)
        S = [1, 1, 0]
        self.E = np.dot(U, np.dot(np.diag(S), V))

        return self.E
    
    def get_E(self):
        return self.E
    
    def draw_line_from_eqn(self, img, arr):
        a, b, c = arr
        x1, y1 = 0, int(-c/b)
        x2, y2 = img.shape[1], int((-c-a*img.shape[1])/b)

        # Draw the line on the image
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    
    def show_epipolar_lines(self, correspondences, img1, img2):
        for point_pair in correspondences:
            img1_pt = point_pair[0]
            img2_pt = point_pair[1]

            img1_pt = np.expand_dims([img1_pt[0], img1_pt[1], 1], 1)
            img2_pt = np.expand_dims([img2_pt[0], img2_pt[1], 1], 1)

            img1_line = np.matmul(self.F, img2_pt)
            img2_line = np.matmul(self.F, img1_pt)

            self.draw_line_from_eqn(img1, img1_line)
            self.draw_line_from_eqn(img2, img2_line)

        cv2.imshow("Epipolar Lines", np.hstack((img1, img2)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
