import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

# TO DO: RESIZE
'''
I should downscale to find corners, then upscale corners to fit original image.
'''


DEVICE_NAME = 'webcam_A_480p'
# DATA_DIR = 'C:\\Users\\JakeC\\OneDrive\\Documents\\GitHub\\ComputerVision\\CameraCalibration\\data'
RELATIVE_DATA_DIR = '.\\data'
CALIBRATION_IMAGE_DIR = os.path.join(RELATIVE_DATA_DIR, 'calibration images', DEVICE_NAME)
INTRINSIC_MATRIX_DIR = os.path.join(RELATIVE_DATA_DIR, 'intrinsic matricies', DEVICE_NAME, 'intrinsic matrix.csv')
TEST_DIR = os.path.join(RELATIVE_DATA_DIR, 'calibration images', DEVICE_NAME + '_Test')

CHESSBOARD_SHAPE = (9,6)
RESIZE_FACTOR = 1 #CHANGE THIS TO A RESOLUTION, NOT A FACTOR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((CHESSBOARD_SHAPE[0] * CHESSBOARD_SHAPE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARD_SHAPE[0],0:CHESSBOARD_SHAPE[1]].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob(os.path.join(CALIBRATION_IMAGE_DIR, '*.png'))
# sharpenKernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

i = 0
for fname in images:
    # print(1)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    downsize = cv2.resize(gray, (0, 0), fx = RESIZE_FACTOR, fy = RESIZE_FACTOR)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(downsize, CHESSBOARD_SHAPE, None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        print('found: ' + str(i))
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(downsize ,corners, (11,11), (-1,-1), criteria) / RESIZE_FACTOR
        imgpoints.append(corners2)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, CHESSBOARD_SHAPE, corners2, ret)
        # cv2.imshow('img', img)
        cv2.imwrite(os.path.join(TEST_DIR,str(i) + '.png'),img)
        # cv2.waitKey(0)
        i += 1
# cv2.destroyAllWindows()
print 
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('Intrinsic matrix of ', DEVICE_NAME, '= ', mtx)
np.savetxt(INTRINSIC_MATRIX_DIR, mtx)