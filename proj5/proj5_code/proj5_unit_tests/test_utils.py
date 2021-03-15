import numpy as np
import cv2
from proj5_code.utils import *

def test_get_world_vertices():
    '''
    test the get_world_vertices()
    '''
    width=0.4
    height=0.4
    depth=1
    points3d_detect = get_world_vertices(width, height, depth)
    points3d_expect = np.array([[0,0,0], \
                               [0.4,0,0],\
                               [0,0,1],\
                               [0.4,0,1],\
                               [0,0.4,0],\
                               [0.4,0.4,0],\
                               [0,0.4,1],\
                               [0.4,0.4,1]])

    assert(np.allclose(points3d_expect, points3d_detect, atol=0.1))




def test_projection_2d_to_3d():
  '''
  Test projection_2d_to_3d
  '''
  K = np.array([[ 500,   0, 535],
                            [   0, 500, 390],
                            [   0,   0,  -1]])

  R = np.array([[ 0.5,   -1,  0],
                            [   0,    0, -1],
                            [   1,  0.5,  0]])

  t = np.array([[   1,    0, 0, 300],
                              [   0,    1, 0, 300],
                              [   0,    0, 1,  30]])
  P = np.matmul(K, np.matmul(R, t))
  pose2d = np.array([[100,200],[200,300],[300,400]])
  depth = 1
  n=len(pose2d)

  pose3d_detected = projection_2d_to_3d(P, depth, pose2d)
  pose2d_reconstruct = P.dot(np.hstack((pose3d_detected, np.ones((n,1)))).T)[:2,:].T


  assert(np.allclose(pose2d, pose2d_reconstruct, atol=0.1))
