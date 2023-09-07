import csv
import cv2
import numpy as np
from IPython import embed

class FeatureUtils:
    def __init__(self):
        self.n_features = []
        self.matching_files = ['matching1.txt', 'matching2.txt', 'matching3.txt', 'matching4.txt']
        self.match_file_idxes = [1, 2, 3, 4]
        self.matches = {}
        self.total_matches = 0
        # self.inliers = {}

    def read_matching_files(self, base_path):
        for image_i_idx, match_file in zip(self.match_file_idxes, self.matching_files):
            with open(base_path+match_file, 'r') as file:
                reader = csv.reader(file, delimiter=' ')
                for row_idx, row in enumerate(reader):
                    if(row_idx==0):
                        self.n_features.append(int(row[1]))
                        continue
                    n_matches = int(row[0])
                    self.total_matches += n_matches - 1
                    n = 1
                    image_i_u = float(row[4])
                    image_i_v = float(row[5])
                    j_idx = 6
                    while(n<n_matches):
                        image_j_idx = int(row[j_idx])
                        image_j_u = float(row[j_idx+1])
                        image_j_v = float(row[j_idx+2])

                        if self.matches.get((image_i_idx, image_j_idx)) is not None:
                            self.matches[(image_i_idx, image_j_idx)].append([(image_i_u, image_i_v), (image_j_u, image_j_v)])
                        else:
                            self.matches[(image_i_idx, image_j_idx)] = [[(image_i_u, image_i_v), (image_j_u, image_j_v)]]

                        n += 1
                        j_idx += 3
        
        return self.matches

    def plot_matches(self, imga, imgb, matches, window_name="Matches"):
        output_image = np.concatenate((imga, imgb), axis=1)
        h, w, l = imga.shape
        for match in matches:
            imga_pnt = (int(match[0][0]), int(match[0][1]))
            imgb_pnt = (int(match[1][0]+w), int(match[1][1]))
            cv2.circle(output_image, imga_pnt, 2, [255,0,0], -1)
            cv2.circle(output_image, imgb_pnt, 2, [255,0,0], -1)
            cv2.line(output_image, imga_pnt, imgb_pnt, (0,255,0), 1)

        cv2.imshow(window_name, output_image)
        cv2.waitKey(0)

    # def add_inliers(self, img_idxs, point_pair):
    #     image_i_idx, image_j_idx = img_idxs
    #     image_i_u, image_i_v = point_pair[0]
    #     image_j_u, image_j_v = point_pair[1]
    #     if self.inliers.get((image_i_idx, image_j_idx)) is not None:
    #         self.inliers[(image_i_idx, image_j_idx)].append([(image_i_u, image_i_v), (image_j_u, image_j_v)])
    #     else:
    #         self.inliers[(image_i_idx, image_j_idx)] = [[(image_i_u, image_i_v), (image_j_u, image_j_v)]]

