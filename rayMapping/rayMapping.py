import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

DEVICE_NAME = 'webcam_A_480p'
DATA_DIR = '.\\data'
IMAGE_DIR = os.path.join(DATA_DIR, 'test images')
INTRINSIC_MATRIX_DIR = os.path.join(DATA_DIR, 'intrinsic matricies')

IMAGE_NAME = DEVICE_NAME + '_sample_0.png'
POINTS_NAME = DEVICE_NAME + '_points_0.csv'
INTRINSIC_MATRIX_NAME = 'intrinsic matrix.csv'

paths = {
'IMAGE_PATH': os.path.join(IMAGE_DIR, IMAGE_NAME),
'POINTS_PATH': os.path.join(IMAGE_DIR, POINTS_NAME),
'INTRINSIC_MATRIX_PATH': os.path.join(INTRINSIC_MATRIX_DIR, DEVICE_NAME, INTRINSIC_MATRIX_NAME)
}

CIRCLE_RADIUS = 2
CIRCLE_THICKNESS = 2 
COLOR_MAPPING = {114:(0,0,255), 103:(0,255,0), 98:(255,0,0), 111:(0,128,255), 109:(255,0,255), 99:(255,255,0), 121:(0,255,255), 66:(0,0,0)}

CAMERA_RESOLUTION = (640,480)



def plotPoints(img:np.array, points:np.array):

    for point in points:
        img = cv2.circle(img, (int(point[2]),int(point[3])), CIRCLE_RADIUS, COLOR_MAPPING[point[1]], CIRCLE_THICKNESS)
    # cv.imshow('img', img)
    # cv.waitKey(0)
    return img

def convertToHomogenous(u:np.array, min:int = 0, max:int = 2, numPoints:int = 11, resolution:tuple = CAMERA_RESOLUTION):
    
    if len(u.shape) == 1:
        u = np.expand_dims(u, axis = 0)

    U = np.zeros((u.shape[0], numPoints, 3))
    for u_idx, point in enumerate(u):
        for point_idx in range(numPoints):
            w = point_idx * (max - min) / (numPoints - 1) 
            U[u_idx][point_idx][0] = (point[2] - (resolution[0] // 2)) * w
            U[u_idx][point_idx][1] = (point[3] - (resolution[1] // 2)) * w
            U[u_idx][point_idx][2] = w
    
    
    return U #np.squeeze(U)

def convertPixel2CameraCoords(u:np.array, 
                              M_int:np.array,
                              displayImg:np.array,
                              min:int = 0, 
                              max:int = 2, 
                              numPoints:int = 11, 
                              resolution:tuple = CAMERA_RESOLUTION,
                              ):
    if len(u.shape) == 1:
        u = np.expand_dims(u, axis = 0)

    u_w = np.linspace(min, max, numPoints)
    inv_M_int = np.linalg.inv(M_int)
    
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    U = np.zeros((u.shape[0], numPoints, 3))
    for u_idx, point in enumerate(u):
        u_u = (point[2]) * u_w
        u_v = (point[3]) * u_w
        [x,y,z] = np.matmul(inv_M_int, [u_u,u_v,u_w])


        ax.plot3D(x, y, z)
        # for point_idx in range(numPoints):
        #     w = point_idx * (max - min) / (numPoints - 1) 
        #     U[u_idx][point_idx][0] = (point[2] - (resolution[0] // 2)) * w
        #     U[u_idx][point_idx][1] = (point[3] - (resolution[1] // 2)) * w * -1
        #     U[u_idx][point_idx][2] = w
    ax.imshow(displayImg,)
    plt.show()

image = cv2.imread(paths['IMAGE_PATH'])
points = np.loadtxt(paths['POINTS_PATH'])
# print(points)
img = plotPoints(image, points)

def getRotationMatrix(point:np.array = np.array([1,0,0]), 
                      destination:np.array = np.array([1,0,0]),
                      origin:np.array = np.array([0,0,0]),                      
                      ):
    vector0 = (point - origin) / np.linalg.norm(point - origin)
    vector1 = (destination - origin) / np.linalg.norm(destination - origin)
    axis = np.cross(vector0, vector1)
    cos_theta = np.dot(vector0, vector1)
    sin_theta = np.linalg.norm(axis)
    axis = axis/ sin_theta
    x, y, z = axis

    M = np.array([
        [(cos_theta + x*x*(1-cos_theta)), (x*y*(1-cos_theta) - z*sin_theta), (x*z*(1-cos_theta) + y*sin_theta)],
        [(z*x*(1-cos_theta) + z*sin_theta), (cos_theta + y*y*(1-cos_theta)), (y*z*(1-cos_theta) - x*sin_theta)],
        [(z*x*(1-cos_theta) - y*sin_theta), (z*y*(1-cos_theta) + x*sin_theta), cos_theta + z*z*(1-cos_theta)]
    ])
    return M


intrinsicMatrix = np.loadtxt(paths['INTRINSIC_MATRIX_PATH'])

convertPixel2CameraCoords(u = points, M_int= intrinsicMatrix, displayImg=img)

