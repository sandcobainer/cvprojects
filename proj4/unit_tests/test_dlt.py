"""Unit tests for the DLT part of the project."""

import unittest

import numpy as np
from proj4_code.dlt import (generate_homogenous_system,
                            get_eigenvector_with_smallest_eigenvector)


def test_generate_homogenous_system():
    """Unit test for generate_homogenous_system()"""
    pts2d = np.array([
        [20.0, -30.0],
        [70.0, 50.0],
        [100.0, 124.8],
    ])

    pts3d = np.array([
        [50.0, 70.0, 20.0],
        [70.0, 50.0, -30.0],
        [120.0, -10.0, 50.0],
    ])

    expected = np.array([
        # first point
        [0.0,  0.0,  0.0,  0.0,
         -50.0, - 70.0, -20.0, -1.0,
         -1500.0, -2100.0, -600.0, -30.0],
        [50.0, 70.0, 20.0, 1.0,
         0.0,  0.0,  0.0, 0.0,
         -1000.0, -1400.0, -400.0, -20.0],
        # second point
        [0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         -7.0000e+01, -5.0000e+01, 3.0000e+01, -1.0000e+00,
         3.5000e+03,  2.5000e+03, -1.5000e+03,  5.0000e+01],
        [7.0000e+01,  5.0000e+01, -3.0000e+01,  1.0000e+00,
         0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,
         -4.9000e+03, -3.5000e+03,  2.1000e+03, -7.0000e+01],
        # third point
        [0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,
         -1.2000e+02, 1.0000e+01, -5.0000e+01, -1.0000e+00,
         1.4976e+04, -1.2480e+03, 6.2400e+03, 1.2480e+02],
        [1.2000e+02, -1.0000e+01, 5.0000e+01, 1.0000e+00,
         0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,
         -1.2000e+04,  1.0000e+03, -5.0000e+03, -1.0000e+02]]



    )

    computed = generate_homogenous_system(pts2d, pts3d)

    np.testing.assert_allclose(computed, expected, rtol=1e-3)


def test_get_eigenvector_with_smallest_eigenvector():
    """Unit test for get_eigenvector_with_smallest_eigenvector"""

    np.random.seed(0)

    A = np.random.randn(7, 13)

    computed = get_eigenvector_with_smallest_eigenvector(A)

    computed_vec_norm = np.linalg.norm(computed)

    mul_norm = (np.linalg.norm(A @ computed) / computed_vec_norm)

    np.testing.assert_allclose(0, mul_norm, atol=1e-5)
