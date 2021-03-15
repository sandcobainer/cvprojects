import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
from proj4_code.projection_matrix import decompose_camera_matrix


def transformation_matrix(wRc_T, wtc):
    """
    Compute the transformation matrix that transform points in the world 
    coordinate system to camera coordinate system.
    
    Args:
    - wRc_T: 3x3 orthonormal rotation matrix (numpy array)
    - wtc: A numpy array of shape (1, 3) representing the camera center
           location in world coordinates
              
    Returns:
    - M: 4x4 transformation matrix that transform points in the world 
         coordinate system to camera coordinate system.
    """
    M = np.zeros((4,4))
    M[:3, :3] = wRc_T
    M[:3, 3] = np.transpose( np.matmul(-wRc_T, np.transpose(wtc)) )
    M[3,3] = 1
    return M


def convert_3d_points_to_camera_coordinate(M, points_3d_w):
    """
    Transform points in the world coordinate system to camera coordinate 
    system using the transformation matrix.
    
    Args:
    - M: 4x4 transformation matrix that transform points in the world 
         coordinate system to camera coordinate system.
    - points_3d_w: n x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                   or n x 3 array of points [X_i,Y_i,Z_i]. Your code needs to take
                   care of both cases.
         
    Returns:
    - points_3d_c: n x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates.
    """
    n = points_3d_w.shape[0]
    points_3d_c = np.zeros((n,4))
    if (points_3d_w.shape[1] == 3):
        points_3d_w = np.hstack( (points_3d_w, np.ones((n,1)) ) )
    
    for i in range(0, n): # n
        points_3d_c[i,:] = np.matmul( M, np.transpose( points_3d_w[i,:]))
    print(points_3d_c)
    return points_3d_c


def projection_from_camera_coordinates(K, points_3d_c):
    """
    Args:
    -  K: 3x3 matrix decomposed from projection matrix K.
    -  points_3d_c : n x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                     or n x 3 array of points [X_i,Y_i,Z_i], which should be the 
                     coordinates of the bounding box's eight vertices in camera 
                     coordinate system.
    Returns:
    - projected_points_2d : n x 2 array of points in non-homogenous image coordinates
    """
    # normalize K
    K /= K[-1, -1]
    n = points_3d_c.shape[0]
    projected_points_2d = np.zeros((n,2))
    if (points_3d_c.shape[1] == 3):
       points_3d_c = np.hstack( (points_3d_c, np.ones((n,1)) ) )
    
    I0 = np.hstack( (np.identity(3), np.zeros((3,1)) ) )
    
    for i in range(0, n): # n
        uvw = np.matmul (np.matmul(K, I0), np.transpose( points_3d_c[i, :]))
        uvw = uvw / uvw[2]    
        projected_points_2d[i,:] = uvw[:-1]
       
    return projected_points_2d


def visualize_bounding_box_camera_coordinates(P, points_3d_w, img):
    """
    Visualize a bounding box over the box-like item in the image.
    
    Args:
    -  P: 3x4 camera projection matrix
    -  points_3d_w : 8 x 4 array of points [X_i,Y_i,Z_i,1] in homogenouos coordinates
                     or 8 x 3 array of points [X_i,Y_i,Z_i], which should be the 
                     coordinates of the bounding box's eight vertices in world 
                     coordinate system.
    -  img: A numpy array, which should be the image in which we are going to 
            visualize the bounding box.
    """
    # find K and cRw from P
    K, wRc_T = decompose_camera_matrix(P)

    M = np.matmul(np.linalg.inv(K), P)
    M = np.concatenate([M, np.array([[0, 0, 0, 1]])], axis=0)

    # transform the vertices to camera coordinates
    points_3d_c = convert_3d_points_to_camera_coordinate(M, points_3d_w)

    # project to 2D
    projected = projection_from_camera_coordinates(K, points_3d_c)

    # load and show the image
    _, ax = plt.subplots()

    ax.imshow(img)

    units = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

    # draw the bounding box
    for i, j in itertools.combinations(range(len(points_3d_w)), 2):
        d = points_3d_w[i, :] - points_3d_w[j, :]
        mod = np.dot(d, d)
        if any(np.square(np.dot(d, unit)) == mod for unit in units):
            ax.plot((projected[i, 0], projected[j, 0]),
                    (projected[i, 1], projected[j, 1]), '-', c='green')
