import numpy as np
import cv2
import glob

def calibrate(path='../data/cali/example/', m=4, n=7):
    ''' 
    Use opencv to calibrate the camera

    Args:
        path: the folder to the calibration pictures
        m,n: the number of grids used for calibration
        (which is not the number of total grids in the cheesebord, but two grids smaller than it)


    Returns: intrinsic_matrix
    '''
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((n*m,3), np.float32)
    objp[:,:2] = np.mgrid[0:n,0:m].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path += '*.jpg'
    print(path)
    images = glob.glob(path)
    # print(images)
    h, w = 0, 0
    gray = None
    for fname in images:
        # img = cv2.imread(fname)
        # cv2.imshow('img',img)
        # cv2.waitKey(500)
        img = cv2.imread(fname)
        h,  w = img.shape[:2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (n,m),None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (n,m), corners2,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    cv2.destroyAllWindows()
    ret, mtx, dist, _, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    intrinsic_matrix, _=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    print(intrinsic_matrix)
    return intrinsic_matrix