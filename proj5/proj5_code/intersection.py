import numpy as np
import cv2


def check_hand_inside_bounding_box(hand, pts):
    """
    This function checks whether the hand is inside the bounding box of the
    chair or not.

    Args:
        hand: 3D coordinate of the hand (numpy.array, size 1*3)
        pts: 3D coordinates of the 8 vertices of the bounding box (numpy.array, size 8*3)

    Returns:
        inside: boolean value, True if hand is inside the bounding box, and
                False otherwise.

    Hints: Remember, in this project, we establish the world frame around the bounding box.
    This assumption could make this problem much easier.
    """
    x_max = np.max(pts[:,0])
    y_max = np.max(pts[:,1])
    z_max = np.max(pts[:,2])
    
    if ( np.all(hand > 0) and (hand[0] <= x_max) and (hand[1] <= y_max) and (hand[2] <= z_max) ):
      return True
    else:
      return False
    

def draw_box_intersection(image, hand, pts, pts_2d):
    """
    Draw the bounding box (in blue) around the chair. If the hand is within the
    bounding box, then we draw it with another color (red)

    Args:
        image: the image in which we'll draw the bounding box, the channel follows RGB order
        hand: 3D coordinate of the hand (numpy.array, 1*3)
        pts: 3D coordinates of the 8 vertices of the bounding box (numpy.array, 8*3)
        pts_2d: 2D coordinates of the 8 vertices of the bounding box (numpy.array, 8*2)
    
    Returns:
        image: annotated image
    """

#    if np.shape(pts)[1] == 3:
#        pts = np.concatenate([pts, np.ones((8,1))], axis=1)

    color = (0, 0, 255)
    if check_hand_inside_bounding_box(hand, pts):
        color = (255, 0, 0)
        print("Check succeed!")

    thickness = 5

    scaleX = image.shape[1]
    scaleY = image.shape[0]

    lines = [(0, 1), (1, 3), (0, 2), (3, 2), (1, 5), (0, 4), (2, 6), (3, 7), (5, 7), (6, 7), (6, 4), (4, 5)]
    for line in lines:
        pt0 = pts_2d[line[0]]
        pt1 = pts_2d[line[1]]
        pt0 = (int(pt0[0] * scaleX), int(pt0[1] * scaleY))
        pt1 = (int(pt1[0] * scaleX), int(pt1[1] * scaleY))
        cv2.line(image, pt0, pt1, color, thickness)


    for i in range(8):
        pt = pts_2d[i]
        pt = (int(pt[0] * scaleX), int(pt[1] * scaleY))
        cv2.circle(image, pt, 8, (0, 255, 0), -1)
        cv2.putText(image, str(i), pt, cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    print(image.shape)
    return image
