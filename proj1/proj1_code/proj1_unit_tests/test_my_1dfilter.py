"""Unit tests for function create_1D_Gaussian_kernel in models.py"""

import torch

from proj1_code.part1 import my_1dfilter


def test_filter_with_box_kernel():
    kernel = torch.tensor([1, 1, 1]).float()

    signal = torch.tensor([-0.5, 0.5, 1, -0.25, 0.75, -0.33, +0.20]).float()

    output = my_1dfilter(signal, kernel)

    expected = torch.tensor([0, 1, 1.25, 1.5, 0.17, 0.62, -0.13]).float()
    
    assert torch.allclose(expected, output)


def test_filter_with_asymmetric_kernel():
    kernel = torch.tensor([-2, 1]).float()

    signal = torch.tensor([-0.5, 0.5, 1, -0.25, 0.75, -0.33, +0.20]).float()

    output = my_1dfilter(signal, kernel)

    expected = torch.tensor(
        [-0.5, 1.5, 0, -2.25,  1.25, -1.83,  0.86]).float()

    assert torch.allclose(expected, output)
