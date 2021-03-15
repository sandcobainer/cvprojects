import numpy as np
import cv2
import matplotlib.pyplot as plt


def perspective_n_points(initial_box_points_3d, box_points_2d, intrinsic_matrix):
    """
    This function calculates the camera pose given 2D feature points 
    with its 3D real world coordiantes matching.

    Args:
    -    initial_box_points_3d: N x 3 array, vertices 3D points in world coordinate
    -    box_points_2d: N x 2 array, vertices 2D points in image coordinate
    -    intrinsic_matrix, 3 x 3 array, the intrinsic matrix of your camera

    Returns:
    -    wRc_T: 3 x 3 array, the rotation matrix that transform from world to camera
    -    camera_center: 3 x 1 array, then camera center in world coordinate
    -    P: 3x4 projection matrix

    Hints: you can use cv2.solvePnP and cv2.Rodrigues in this function
    
    """


    wRc_T = None
    camera_center = None 
    P = None

    initial_box_points_3d = np.array(initial_box_points_3d, dtype='float32')
    box_points_2d  = np.array(box_points_2d, dtype='float32')
    distCoeffs=np.zeros((8,1))

    _, rvec, tvec = cv2.solvePnP(initial_box_points_3d, box_points_2d, intrinsic_matrix, distCoeffs)
    
    wRc_T, _ = cv2.Rodrigues(rvec)
    P = np.matmul(intrinsic_matrix, np.append(wRc_T, tvec, axis=1))
    camera_center = - np.matmul( np.transpose(wRc_T), tvec)
    
    return wRc_T, camera_center, P

def plot_box_and_camera(points_3d, camera_center, R):
    """
    Visualize the actual 3D points and the estimated 3D camera center.

    """
    
    print("The camera center is at: \n", camera_center)

    v1 = R[:, 0]
    v2 = R[:, 1]
    v3 = R[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue',
               marker='o', s=10, depthshade=0)
    camera_center = camera_center.squeeze()
    ax.scatter(camera_center[0],  camera_center[1], camera_center[2], c='red',
               marker='x', s=20, depthshade=0)

    cc0, cc1, cc2 = camera_center
    
    point0 = points_3d[0]
    ax.plot3D([point0[0], point0[0]+2], [point0[1], point0[1]], [point0[2], point0[2]], c='r')
    ax.plot3D([point0[0], point0[0]], [point0[1], point0[1]+2], [point0[2], point0[2]], c='g')
    ax.plot3D([point0[0], point0[0]], [point0[1], point0[1]], [point0[2], point0[2]+2], c='b')

    ax.plot3D([cc0, cc0+v1[0]], [cc1, cc1+v1[1]], [cc2, cc2+v1[2]], c='r')
    ax.plot3D([cc0, cc0+v2[0]], [cc1, cc1+v2[1]], [cc2, cc2+v2[2]], c='g')
    ax.plot3D([cc0, cc0+v3[0]], [cc1, cc1+v3[1]], [cc2, cc2+v3[2]], c='b')


    # draw edges of the box

    min_z = min(points_3d[:, 2])
    min_x = min(points_3d[:, 0])
    min_y = min(points_3d[:, 1])
    for p in points_3d:
        x, y, z = p
        ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c='black', linewidth=1)
        ax.plot3D(xs=[x, min_x], ys=[y, y], zs=[z, z], c='black', linewidth=1)
        ax.plot3D(xs=[x, x], ys=[y, min_y], zs=[z, z], c='black', linewidth=1)
    x, y, z = camera_center
    ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c='black', linewidth=1)


