import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from proj5_code.utils import *

from scipy.ndimage.filters import maximum_filter


import tensorflow as tf

def detect_peak(image, filter_size=3, order=0.5):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image,mask=~(image == local_max))

    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index

def decode(hm, displacements):
    '''
    Decode the heatmap and displacement feilds from the encoder.
    Args:
        hm: heatmap
        displacements: displacement fields

    Returns:
        normalized vertices coordinates in 2D image
    '''
    hm = hm.reshape(hm.shape[2:])     # (40,30)

    peaks=hm.argmax()
    peakX = [peaks%30]
    peakY = [peaks//30]


    peaks=hm.argmax()
    peakX = [peaks%30]
    peakY = [peaks//30]

    scaleX = hm.shape[1]
    scaleY = hm.shape[0]
    objs = []
    for x,y in zip(peakX, peakY):
        conf = hm[y,x]
        print(conf)
        points=[]
        for i in range(8):
            dx = displacements[0, i*2  , y, x]
            dy = displacements[0, i*2+1, y, x]
            points.append((x/scaleX+dx, y/scaleY+dy))
        objs.append(points)
    return objs


def draw_box(image, pts):
    '''
    Drawing bounding box in the image
    Args:
        image: image array
        pts: bounding box vertices

    Returns:

    '''
    scaleX = image.shape[1]
    scaleY = image.shape[0]

    lines = [(0,1), (1,3), (0,2), (3,2), (1,5), (0,4), (2,6), (3,7), (5,7), (6,7), (6,4), (4,5)]
    for line in lines:
        pt0 = pts[line[0]]
        pt1 = pts[line[1]]
        pt0 = (int(pt0[0]*scaleX), int(pt0[1]*scaleY))
        pt1 = (int(pt1[0]*scaleX), int(pt1[1]*scaleY))
        cv2.line(image, pt0, pt1, (255,0,0), thickness=10)

    for i in range(8):
        pt = pts[i]
        pt = (int(pt[0]*scaleX), int(pt[1]*scaleY))
        cv2.circle(image, pt, 8, (0,255,0), -1)
        cv2.putText(image, str(i), pt,  cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)


def inference(img, model_path):
    """
    Running inference given the image model, and generate heatmap and displacements.
    If you don't know what is heatmap and displacement fields, you should go to read the objectron paper.
    (https://arxiv.org/pdf/2003.03522.pdf);
    Besides, the `objectron.py` in the repo 
    Args:
        img: image file
        model_path: .tflite weights file

    Returns: heatmap and displacement files

    """
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']

    input_data = np.zeros(input_shape)
    input_data[0,:,:,0]=img[0,:,:]
    input_data[0,:,:,1]=img[1,:,:]
    input_data[0,:,:,2]=img[2,:,:]
    input_data = np.array(input_data, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data2 = interpreter.get_tensor(output_details[1]['index'])
    # print(output_data)
    output_data_reshape=np.zeros([1,1,40,30])
    output_data_reshape[0,0,:,:]=output_data[0,:,:,0]

    output_data2_reshape=np.zeros([1,16, 40,30])
    for i in range(16):
        output_data2_reshape[0, i,:,:]=output_data2[0,:,:,i]
    return output_data_reshape, output_data2_reshape

def detect_3d_box(img_path):

    '''
        Given an image, this function detects the 3D bounding boxes' 8 vertices of the chair in the image.
        We will only consider one chair in one single image.
        Similar to pose estimation, you're going to use mediapipe to detect the 3D bounding boxes.
        You should try to understand how does the objectron work before trying to finish this function!

        Args:
        -    img_path: the path of the RGB chair image

        Returns:
        -

        boxes: numpy array of 2D points, which represents the 8 vertices of 3D bounding boxes
        annotated_image: the original image with the overlapped bounding boxes

        Useful functions for usage: inference()
    '''

    if os.path.exists('proj5_code/object_detection_3d_chair.tflite'):
        model_path = 'proj5_code/object_detection_3d_chair.tflite'
    elif os.path.exists('../object_detection_3d_chair.tflite'):
        model_path = "../object_detection_3d_chair.tflite"
    elif os.path.exists('../../object_detection_3d_chair.tflite'):
        model_path = "../../object_detection_3d_chair.tflite"

    boxes = None
    hm = None
    displacements = None

    inshapes = [[1, 3, 640, 480]]
    outshapes = [[1, 16, 40, 30], [1, 1, 40, 30]]
    print(inshapes, outshapes)


    if img_path == 'cam':
        cap = cv2.VideoCapture(0)

    while True:
        if img_path == 'cam':
            _, img_orig = cap.read()
        else:
            img_file = img_path
            img_orig = cv2.imread(img_file)


        img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (inshapes[0][3], inshapes[0][2]))
        img = img.transpose((2,0,1))
        image = np.array(img, np.float32)/255.0
        
        hm, displacements = inference(image, model_path)
        # decode inference result
        boxes = decode(hm, displacements) # 0.7 for original script

        # draw bbox
        for obj in boxes:
            draw_box(img_orig, obj)
        return boxes[0], cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)


