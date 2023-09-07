# Monocular-Multi-view-Structure-from-Motion-SFM-with-Bundle-Adjustment
This project focuses on reconstructing a 3D scene from a set of 2D images, a process known as Structure from Motion (SfM). The goal is to understand and implement the steps involved in SfM, inspired by the work of Agarwal et. al in "Building Rome in a Day" and tools like Microsoft Photosynth and VisualSFM.

## Steps Involved in SfM:
- Feature Matching and Outlier rejection using RANSAC
- Estimating Fundamental Matrix
- Estimating Essential Matrix from Fundamental Matrix
- Estimate Camera Pose from Essential Matrix
- Check for Cheirality Condition using Triangulation
- Perspective-n-Point
- Bundle Adjustment

## Dataset
The dataset consists of 5 images of Unity Hall at WPI taken using a Samsung S22 Ultra’s primary camera. The images are already distortion corrected and resized to 800×600px. Keypoint matching data using SIFT keypoints and descriptors is also provided.

## Classical Approach to SfM
1. **Feature Matching, Fundamental Matrix, and RANSAC**  
We use SIFT keypoints and descriptors for keypoint matching. The matches are refined using RANSAC.

2. **Estimating Fundamental Matrix**  
The fundamental matrix relates corresponding points in two images from different views. It is essential to understand epipolar geometry to grasp the concept of the fundamental matrix.

3. **Estimate Essential Matrix from Fundamental Matrix**  
The essential matrix relates the corresponding points in two images, assuming the cameras obey the pinhole model.

4. **Estimate Camera Pose from Essential Matrix**  
The camera pose consists of the rotation and translation of the camera with respect to the world.

5. **Triangulation Check for Cheirality Condition**  
We need to check the cheirality condition to ensure the reconstructed points are in front of the cameras.

6. **Perspective-n-Points**  
Estimate the 6 DOF camera pose using linear least squares.

7. **Bundle Adjustment**  
Refine the camera poses and 3D points together by minimizing reprojection error.

## Implementation
- `Wrapper.py`: This is the main program that runs the full pipeline of SfM based on the above algorithms.
- `EstimateFundamentalMatrix.py`: Estimates the fundamental matrix.
- `GetInlierRANSANC.py`: Estimates inlier correspondences using fundamental matrix-based RANSAC.
- `EssentialMatrixFromFundamentalMatrix.py`: Computes the essential matrix from the fundamental matrix.
- `ExtractCameraPose.py`: Extracts the camera pose from the essential matrix.
- `LinearTriangulation.py` & `NonlinearTriangulation.py`: Implements both linear and non-linear triangulation methods.
- `LinearPnP.py` & `NonlinearPnP.py`: Implements both linear and non-linear PnP methods.
- `BundleAdjustment.py`: Implements the bundle adjustment method.
- `BuildVisibilityMatrix.py`: Constructs the visibility matrix.


## Instructions
1. Clone the repository.
2. Ensure you have all the required libraries and dependencies installed.
3. Run `Wrapper.py` to execute the full pipeline.
4. Compare your results with the provided outputs.

## Results
You can compare your results against the output from Visual SfM. Sample outputs are provided in the figures.

### Initial Feature Matches
![feature_matches](Outputs/feature_matches.png)


### Inlier Features (after RANSAC)
![feature_matches](Outputs/inlier_features_ransac.png)


### Epipoles and Epipolar Lines
![epipolar](Outputs/epipolar_1-2_new.png)


### Linear Triangulation
![Triangulation_1](Outputs/linear_triangulation_1.png)

![Triangulation](Outputs/linear_triangulation.png)

### Non-Linear Triangulation
![Non-Linear-Triangulation](Outputs/non-linear_triangulation.png)

### Final Output
![Multiview structure from motion](https://user-images.githubusercontent.com/63463655/236314227-9ccb62e4-fa8e-4f85-ae54-ef4660ceb949.png)

## References
- Agarwal et. al, "Building Rome in a Day".
- VisualSFM: [Download here](https://user-images.githubusercontent.com/63463655/236314227-9ccb62e4-fa8e-4f85-ae54-ef4660ceb949.png).

