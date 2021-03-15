import numpy as np
import math
from proj4_code.projection_matrix import projection, estimate_camera_matrix
from proj4_code.dlt import estimate_projection_matrix_dlt


def calculate_num_ransac_iterations(prob_success: float,
                                    sample_size: int,
                                    ind_prob_correct: float) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float the probability that each element in a sample is correct

    Returns:
    -   the number of RANSAC iterations needed
    """
    num_iterations = 0

    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    raise NotImplementedError('calculate_num_ransac_iterations in ransac.py needs to be implemented')
    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

    return num_iterations


def find_inliers(points_2d: np.ndarray,
                 points_3d: np.ndarray,
                 P: np.ndarray,
                 threshold: float):
    """Find the inliers' indices for a given model.

    Hint: you can copy some code from evaluate_points function to get the 
    residual error for each point.

    Args:
    -   x_0s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the left image
    -   F: The proposed fundamental matrix
    -   x_1s: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from the right image
    -   threshold: the maximum error for a point correspondence to be
                    considered an inlier
    Each row in x_1s and x_0s is a proposed correspondence (e.g. row #42 of x_0s is a point that
    corresponds to row #42 of x_1s)

    Returns:
    -    inliers: 1D array of the indices of the inliers in x_0s and x_1s

    """

    projected_2d = projection(P, points_3d)

    distances = np.linalg.norm(projected_2d - points_2d, axis=1)

    indices = np.where(distances < threshold)[0]

    return indices


def ransac_projection_matrix(pts2d: np.ndarray,
                             pts3d: np.ndarray,
                             inlier_threshold: float = 8.0,
                             num_iterations: int = 100):
    """Find the projection matrix with RANSAC.

    Use RANSAC to find the best projection matrix by randomly sampling 
    correspondences. You will call your estimate_projection_matrix_dlt from part
    2 of this assignment.



    Tips:
        1. You will want to call your function for solving P with the random
           sample and then you will want to call your function for finding
           the inliers.
        2. You will also need to choose an error threshold to separate your
           inliers from your outliers. We suggest a threshold of 8.
        3. find_inliers has been written for you in this file.


    Args:
        pts2d: numpy array of shape nx2, with the 2D measurements.
        pts3d: numpy array of shape nx3, with the actual 3D coordinates.

    Returns:
        best_P: A numpy array of shape (3, 4) representing the best fundamental
                projection matrix estimation.
        inliers_pts2d: A numpy array of shape (M, 2) representing the subset of
                       points from pts2d which are inliers w.r.t. best_P.
        inliers_pts3d: A numpy array of shape (M, 3) representing the subset of
                       points from pts3d which are inliers w.r.t. best_P.

    """

    num_input_points = pts2d.shape[0]

    best_P = np.random.rand(3, 4)
    best_inlier_count = 0
    inliers_pts2d = np.array([])
    inliers_pts3d = np.array([])

    for _ in range(num_iterations):
        # randomly sample 6 correspondences
        idxes = np.random.choice(num_input_points, size=6, replace=False)

        sampled_pts2d = pts2d[idxes]
        sampled_pts3d = pts3d[idxes]

        ########################################################################
        # TODO: YOUR CODE HERE
        ########################################################################

        P_initial = None
        P_sample = None
        inliers = []
        
        raise NotImplementedError('ransac_projection_matrix in ransac.py needs to be implemented')

        ########################################################################
        #                             END OF YOUR CODE
        ########################################################################

        if len(inliers) > best_inlier_count:
            best_inlier_count = len(inliers)
            inliers_pts2d = pts2d[inliers]
            inliers_pts3d = pts3d[inliers]
            best_P = P_sample

    print('Found projection matrix with support ', best_inlier_count)

    return best_P, inliers_pts2d, inliers_pts3d
