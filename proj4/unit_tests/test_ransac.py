import pytest
import numpy as np
from proj4_code import ransac


def test_calculate_num_ransac_iterations():
    Fail = False
    data_set = [(0.99, 1, 0.99, 1),
                (0.99, 10, 0.9, 11),
                (0.9, 15, 0.5, 75450),
                (0.95, 5, 0.66, 22)]

    for prob_success, sample_size, ind_prob, num_samples in data_set:
        S = ransac.calculate_num_ransac_iterations(
            prob_success,
            sample_size,
            ind_prob
        )
        assert pytest.approx(num_samples, abs=1.0) == S
