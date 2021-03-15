#!/usr/bin/python3

import numpy as np

from proj4_code.camera_coordinates import (
    transformation_matrix,
    convert_3d_points_to_camera_coordinate,
    projection_from_camera_coordinates,
)


def verify(function) -> str:
  """ Will indicate with a print statement whether assertions passed or failed
    within function argument call.

    Args:
    - function: Python function object

    Returns:
    - string
  """
  try:
    function()
    return "\x1b[32m\"Correct\"\x1b[0m"
  except AssertionError:
    return "\x1b[31m\"Wrong\"\x1b[0m"


def test_transformation_matrix():
    '''
        tests whether projection was implemented correctly
    '''

    test_wRc_T = np.array([[ 0.5,   -1,  0],
                      [   0,    0, -1],
                      [   1,  0.5,  0]])
    test_wtc = np.array([[30, 30, -30]])
    test_M = np.array([[0.5,  -1.,    0.,   15.],
                       [0.,    0.,   -1.,  -30.],
                       [1.,    0.5,   0.,  -45.],
                       [0.,    0.,    0.,    1.]])

    M = transformation_matrix(test_wRc_T, test_wtc)
    assert M.shape == test_M.shape
    assert np.allclose(M, test_M, atol=1e-8)


def test_convert_3d_points_to_camera_coordinate():
    '''
        tests whether the objective function has been implemented correctly
        by comparing fixed inputs and expected outputs
    '''

    test_M = np.array([[0.5, -1.,   0.,   0.],
                       [0.,   0.,  -1.,   1.],
                       [1.,   0.5,  0.,  -5.],
                       [0.,   0.,   0.,   1.]])

    test_3D = np.array([[311.49450897, 307.88750897,  28.83350897],
                        [306.29458458, 312.14758458,  30.85458458],
                        [307.69425074, 312.35825074,  30.41825074],
                        [308.91388726, 305.95088726,  28.06288726]])

    points_3d_c = np.array([[-152.14025448,  -27.83350897,  460.43826346,    1.],
                            [-159.00029229,  -29.85458458,  457.36837687,    1.],
                            [-158.51112537,  -29.41825074,  458.87337611,    1.],
                            [-151.49394363,  -27.06288726,  456.88933089,    1.]])

    output = convert_3d_points_to_camera_coordinate(test_M, test_3D)
    assert output.shape == points_3d_c.shape
    assert np.allclose(output, points_3d_c, atol=1e-8)

    # test when the points_3d_w is n x 4
    test_3D_2 = np.array([[91.07627562, 109.9176866,  29.35085811,   1.],
                          [69.02369313, 107.85130302, 112.26770525,   1.],
                          [26.47864855,  68.53406823,  97.77880687,   1.],
                          [91.37710693, 111.71196512,  51.94695837,   1.],
                          [95.66973158,  60.95000739,  32.42326924,   1.]])
    
    points_3d_c_2 = np.array([[-64.37954879,  -28.35085811,  141.03511892,    1.],
                              [-73.33945646, -111.26770525,  117.94934464,    1.],
                              [-55.29474395,  -96.77880687,   55.74568266,    1.],
                              [-66.02341165,  -50.94695837,  142.23308949,    1.],
                              [-13.1151416,  -31.42326924,  121.14473527,    1.]])
    
    output2 = convert_3d_points_to_camera_coordinate(test_M, test_3D_2)
    assert output2.shape == points_3d_c_2.shape
    assert np.allclose(output2, points_3d_c_2, atol=1e-8)

def test_projection_from_camera_coordinates():
    '''
        tests whether projection was implemented correctly
    '''

    test_3D = np.array([[311.49450897, 307.88750897,  28.83350897],
                        [306.29458458, 312.14758458,  30.85458458],
                        [307.69425074, 312.35825074,  30.41825074],
                        [308.91388726, 305.95088726,  28.06288726]])

    test_2D = np.array([[1870.48179409, 1851.71716851],
                        [1739.05546169, 1767.5099041],
                        [1767.31728447, 1790.31663462],
                        [1901.18730157, 1885.34965821]])

    dummy_matrix = np.array([[150.,   0., 250.],
                             [0., 150., 250.],
                             [0.,   0.,   1.]])

    projected_2D = projection_from_camera_coordinates(dummy_matrix, test_3D)
    assert projected_2D.shape == test_2D.shape
    assert np.allclose(projected_2D, test_2D, atol=1e-8)

    # test when points_3d_c is n x 4
    test_3D_2 = np.array([[107.28151455,  77.71140755, 147.11745069,   1.],
                          [82.63743171, 210.74206773, 127.59455946,   1.],
                          [129.34969378, 237.08291477, 157.21640484,   1.],
                          [122.59184938,  59.51819274, 239.43391425,   1.],
                          [225.1574657, 190.29581096, 114.76421954,   1.],
                          [87.53969339, 243.22310961,  60.24171087,   1.]])
    test_2D_2 = np.array([[359.38353749, 329.23404788],
                          [347.14845844, 497.74810379],
                          [373.41240144, 476.20054983],
                          [326.80105579, 287.28681853],
                          [544.28701724, 498.72187306],
                          [467.97113361, 855.61803304]])
    projected_2D_2 = projection_from_camera_coordinates(dummy_matrix, test_3D_2)
    assert projected_2D_2.shape == test_2D_2.shape
    assert np.allclose(projected_2D_2, test_2D_2, atol=1e-8)