import os
import torch
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2
from proj5_code.utils import *


def hand_pose_img(test_img):
    """
        Given an image, it calculates the pose of human in the image. 
        To make things easier, we only consider one people on a single image. 
        Pose estimation is actually a difficult problem, in this function, you are going to use
        mediapipe to do this work. You can find more about mediapipe from its official website
        https://google.github.io/mediapipe/solutions/pose#overview
        
        Args:
        -    img: path to rgb image
        
        Returns:
        -    landmark: numpy array of size (n, 2) the landmark detected by mediapipe,
        where n is the length of landmark, 2 represents x and y coordinates
        (Note, not in the range 0-1, you need to get the real 2D coordinates in images)

        the order of these landmark should be consistent with the original order returned by mediapipe
        -    annotated_image: the original image overlapped with the detected landmark
        
        Useful functions/class: mediapipe.solutions.pose, mediapipe.solutions.drawing_utils
    """

    landmark = None
    annotated_image = None

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5)
    image = cv2.imread(test_img)
    # Convert the BGR image to RGB before processing.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rows = image.shape[0]
    cols = image.shape[1]

    results = pose.process(image)
    landmark = results.pose_landmarks

    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
      annotated_image, landmark, mp_pose.POSE_CONNECTIONS)
    pose.close()
    landmark1=np.zeros((len(results.pose_landmarks.landmark),2))
    for i in range(len(results.pose_landmarks.landmark)):
        landmark1[i,:] = [results.pose_landmarks.landmark[i].x*cols, results.pose_landmarks.landmark[i].y*rows]
    return landmark1, annotated_image

