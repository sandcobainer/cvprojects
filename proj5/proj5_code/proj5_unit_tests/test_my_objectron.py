import os
from PIL import Image
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import cv2

from proj5_code.utils import *
from proj5_code.my_objectron import *
from proj5_code.pose_estimate import *


def test_my_objectron():
  '''
  Tests the pose estimate
  '''

  if os.path.exists('data/chair.jpg'):
      test_img = 'data/chair.jpg'

  elif os.path.exists('../../data/chair.jpg'):
      test_img = '../../data/chair.jpg'
  else:
      test_img = '../data/chair.jpg'
  bounding_boxes, annotated_img = detect_3d_box(test_img)
  expected_bb0 = np.array([0.3183, 0.6371])
  detected_bb0 = np.array(bounding_boxes[0])
  assert(np.allclose(expected_bb0, detected_bb0, atol=0.1))

if __name__=="__main__":
  test_my_objectron()
