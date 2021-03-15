import numpy as np
from proj5_code.intersection import *


def test_check_hand_inside_bounding_box():
    """
    test check_hand_inside_bounding_box
    """
    test_box = np.array([[0.,0.,0.],
                         [0.,0.,1.],
                         [0.,2.,0.],
                         [3.,0.,0.],
                         [0.,2.,1.],
                         [3.,0.,1.],
                         [3.,2.,0.],
                         [3.,2.,1.]])
    test_hand1 = np.array([1.5,1.,.5])
    test_hand2 = np.array([3.,2.,1.])
    test_hand3 = np.array([-3., 2., 1.])
    assert check_hand_inside_bounding_box(test_hand1, test_box)
    assert check_hand_inside_bounding_box(test_hand2, test_box)
    assert not check_hand_inside_bounding_box(test_hand3, test_box)