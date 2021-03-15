import numpy as np

from proj5_code.pnp import *

def test_perspective_n_points():
  '''
  Test perspective_n_points
  '''
  K = np.array([[ 480,   0, 240],
                [   0, 640, 320],
                [   0,   0,  1]])


  box_2d = np.array([
                    [152.71358728, 407.62768173],
                    [126.6287899,  543.62051392],
                    [116.25713825,  68.85868454],
                    [ 62.28662968,  17.68793488],
                    [304.58366632, 395.42098427],
                    [365.79633236, 525.12167358],
                    [320.96123457,  60.44083023],
                    [409.12731171,  14.99428177]])

  box_3d = np.array([
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.0, 0.0, 1.5],
                    [0.5, 0.0, 1.5],
                    [0.0, 0.5, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 1.5],
                    [0.5, 0.5, 1.5]])

  expect_rotation = np.array([[ 0.08957669,  0.99577521, -0.02019296],
                              [ 0.56737769, -0.06768155, -0.82067153],
                              [-0.81857105,  0.062056,   -0.57104333]])

  expect_trans = np.array([[1.75812284],
                           [0.15824511],
                           [1.75326393]])

  expected_P = np.array([[-1.53460231e+02,  4.92865541e+02, -1.46743027e+02,  4.49087649e+02],
                        [ 1.01178987e+02, -2.34582792e+01, -7.07963644e+02,  1.06707419e+03],
                        [-8.18571052e-01,  6.20560132e-02, -5.71043330e-01,  2.43051806e+00]]) 
  
  actual_rot, actual_trans, P = perspective_n_points(box_3d, box_2d, K)

  assert(np.allclose(expect_rotation, actual_rot, atol=0.1))
  assert(np.allclose(expect_trans, actual_trans, atol=0.1))
  assert(np.allclose(expected_P, P, atol=0.1))

