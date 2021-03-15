import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def imshow1(img):
    fig, axs = plt.subplots()
    axs.imshow(img)
    axs.axis('off')
    plt.show()


    
                        
                        
def projection_2d_to_3d(P, depth, pose2d):
    """
    This function calculates the inverse projection from 2D feature points to 3D real world coordiantes.

    Args:
    -    P: size is (3,4), camera projection matrix which combines both camera pose and intrinsic matrix
    -    depth: scalar, which provides the depth information (physica distance between you and camera in real world), in meter
    -    pose2d, size (n,2), where n is the number of 2D pose feature points in (x,y) image coordinates

    Returns:
    -    pose3d, size (n,3), where n is the number of 2D pose points. These are the 3D real-world
    coordiantes of human pose in the chair frame

    Hints:
    When only one 2D point is considered, one very easy way to solve this is treating it
    as three equations with three unknowns. However, since this is a linear system,
    it can also be solve via matrix manipulation. You can try to treat the P as a 3*3 matrix plus a 3*1 column vector,
    and see if it helps
    """
    pose3d = np.zeros((pose2d.shape[0],3))

    P_3x3 = P[:, :3]
    P_3x1 = P[:,3]

    for n in range(0, pose2d.shape[0]):
      point_3d = np.append(pose2d[n, :], np.ones(1), axis=0)
      pose3d[n,:] = np.matmul( np.linalg.inv(P_3x3) , ((depth* point_3d) - P_3x1) )
    
    return pose3d


def get_world_vertices(width, height, depth):
    """
    Given the real size of the chair, return the real-world coordinates of the eight vertices
    in the same order as the detected bounding box from part 1
    Args:
        width: width of the chair, from vertex 0 to vertex 1
        height: height of the chair, from vertex 0 to vertex 4
        depth: depth of the chair, from vertex 0 to vertex 2
    Returns:
        vertices_world: (8,3), 8 vertices' real-world coordinates (x,y,z)
    """
    vertices_world=np.zeros((8,3))
    vertices_world[0,:] = [0,0,0]
    vertices_world[1,:] = [width, 0, 0]
    vertices_world[2,:] = [0,0,depth]
    vertices_world[3,:] = [width,0,depth]
    vertices_world[4,:] = [0,height,0]
    vertices_world[5,:] = [width,height,0]
    vertices_world[6,:] = [0,height,depth]
    vertices_world[7,:] = [width,height,depth]

    return vertices_world

