"""
PyTorch tutorial on data loading & processing:
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import os
from typing import List, Tuple

import numpy as np
import PIL
import torch
import torch.utils.data as data
import torchvision


def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    """
    Creates a dataset of paired images from a directory.

    The dataset should be partitioned into two sets: one contains images that
    will have the low pass filter applied, and the other contains images that
    will have the high pass filter applied.

    Args
    - path: string specifying the directory containing images
    Returns
    - images_a: list of strings specifying the paths to the images in set A,
        in lexicographically-sorted order
    - images_b: list of strings specifying the paths to the images in set B,
        in lexicographically-sorted order
    """
    images_a = []
    images_b = []
    for f in sorted(os.listdir(path)):
      if(f[1]=='a'):
        images_a.append(f)
      elif(f[1] =='b'):
        images_b.append(f)

    return images_a, images_b


def get_cutoff_standardddeviations(path: str) -> List[int]:
    """
    Gets the cutoff standard deviations corresponding to each pair of images.

    The cutoff are the values you discovered from experimenting in
    part 2.

    Args
    - path: string specifying the path to the .txt file with cutoff standard
      deviation values
    Returns
    - List[int]. The array should have the same
      length as the number of image pairs in the dataset
    """

    cutoffs = []
    with open(path, 'r') as f:
      for line in f:
        cutoffs.append(int(line))

    return cutoffs


class HybridImageDataset(data.Dataset):
    """Hybrid images dataset."""

    def __init__(self, image_dir: str, cf_file: str) -> None:
        """
        HybridImageDataset class constructor.

        You must replace self.transform with the appropriate transform from
        torchvision.transforms that converts a PIL image to a torch Tensor. You can
        specify additional transforms (e.g. image resizing) if you want to, but
        it's not necessary for the images we provide you since each pair has the
        same dimensions.

        Args:
        - image_dir: string specifying the directory containing images
        - cf_file: string specifying the path to the .txt file with cutoff
          standard deviation values
        """
        self.images_a, self.images_b = make_dataset(image_dir)
        self.image_dir = image_dir
        self.cutoffs = get_cutoff_standardddeviations(cf_file)
        self.transform = torchvision.transforms.ToTensor()
                         

    def __len__(self) -> int:
        """Returns number of pairs of images in dataset."""
        if (len(self.images_a) == len(self.images_b)):
          return len(self.images_a)
        else:
          return 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns the pair of images and corresponding cutoff standard deviation
        value at index `idx`.

        Since self.images_a and self.images_b contain paths to the images, you
        should read the images here and normalize the pixels to be between 0 and 1.
        Make sure you transpose the dimensions so that image_a and image_b are of
        shape (c, m, n) instead of the typical (m, n, c), and convert them to
        torch Tensors.

        If you want to use a pair of images that have different dimensions from
        one another, you should resize them to match in this function using
        torchvision.transforms.

        Args
        - idx: int specifying the index at which data should be retrieved
        Returns
        - image_a: Tensor of shape (c, m, n)
        - image_b: Tensor of shape (c, m, n)
        - cutoff: int specifying the cutoff standard deviation corresponding to
          (image_a, image_b) pair

        HINTS:
        - You should use the PIL library to read images
        - You will use self.transform to convert the PIL image to a torch Tensor
        """

        image_a = torch.Tensor()
        image_b = torch.Tensor()
        cutoff = self.cutoffs[idx]
        pilimage_a = PIL.Image.open(os.path.join(self.image_dir, self.images_a[idx]))
        pilimage_b = PIL.Image.open(os.path.join(self.image_dir, self.images_b[idx]))
        
        image_a = self.transform(pilimage_a)
        image_b = self.transform(pilimage_b)
        

        return image_a, image_b, cutoff
