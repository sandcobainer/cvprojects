import os
from PIL import Image
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import torch
import cv2
from proj5_code.my_objectron import *
from proj5_code.pnp import *
from proj5_code.calibration import *
from proj5_code.intersection import *
from proj5_code.pose_estimate import *
from proj5_code.utils import *


def process_video(path):
    """
    This function will process the video that you take and should output a video
    that shows you interacting with one or two chairs and their bounding boxes changing colors.

    Args:
        path: a path to the your video file

    Returns:
        none (But a video file should be generated)

    The recommended approach is to process your video mp4 using cv2.VideoCapture.
    For usage you can look up the official opencv documentation.
    You can split up your video into individual frames, and process each frame
    like we did in the notebook, with the correct parameters and correct calibration.
    These individual frames can be turned back into a video, which you can save to your
    computer.

    A simple tutorial can be found here:
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    """

    inside = None
    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################

    ############################################################################
    #                             END OF YOUR CODE
    ############################################################################

