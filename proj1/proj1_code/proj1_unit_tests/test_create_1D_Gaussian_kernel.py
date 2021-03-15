"""Unit tests for function create_1D_Gaussian_kernel in models.py"""

import torch

from proj1_code.models import create_1D_Gaussian_kernel


def test_tensor_datatype():
    standard_deviation = 1

    computed_kernel = create_1D_Gaussian_kernel(standard_deviation)

    assert computed_kernel.dtype == torch.float32


def test_create_kernel_with_sigma_int():
    standard_deviation = 2

    computed_kernel = create_1D_Gaussian_kernel(standard_deviation)

    expected_kernel = torch.tensor([0.02763055,
                                    0.06628225,
                                    0.12383154,
                                    0.18017382,
                                    0.20416369,
                                    0.18017382,
                                    0.12383154,
                                    0.06628225,
                                    0.02763055]).float()

    assert torch.allclose(expected_kernel, computed_kernel)


def test_kernel_sum():
    computed_kernel = create_1D_Gaussian_kernel(30)

    assert torch.allclose(torch.tensor([1]).float(), torch.sum(computed_kernel))
