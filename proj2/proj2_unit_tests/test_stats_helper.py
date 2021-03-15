import os

import numpy as np

from proj2_code.stats_helper import compute_mean_and_std


def test_mean_and_variance():
  if os.path.exists('proj2_code/proj2_unit_tests/small_data/'):
  	mean, std = compute_mean_and_std('proj2_code/proj2_unit_tests/small_data/')
  else:
  	mean, std = compute_mean_and_std('../proj2_code/proj2_unit_tests/small_data/')
  	
  assert np.allclose(mean, np.array([0.46178914]))
  assert np.allclose(std, np.array([0.256041]))
