import glob
import os
from typing import Tuple
import math
import numpy as np
from PIL import Image, ImageOps
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then scale to [0,1] before computing
  mean and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None
  imageList = list()
  for (dp, dn, fn) in os.walk(dir_name):
    imageList += [os.path.join(dp, file) for file in fn]

  scaler = StandardScaler()
  minmax = MinMaxScaler()

  for idx, img in enumerate(imageList):
    # get grayscaled np image
    pil_image = Image.open(img)
    np_image = np.array(ImageOps.grayscale(pil_image))
    
    np_image = np_image.flatten()
    np_image = np_image / 255.0
    np_image = np_image.reshape(len(np_image),1)

    scaler.partial_fit(np_image)

  mean = scaler.mean_[0]
  std = math.sqrt(scaler.var_[0])
  print(mean,std)
  return mean, std
