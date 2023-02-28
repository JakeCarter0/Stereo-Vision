import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

DEVICE_NAME = 'webcam_A_480p'
TEST_IMAGE_DIR = '.\\data\\test images'
IMG_NAME = DEVICE_NAME + '_sample_0.png'
SAVE_NAME = DEVICE_NAME + '_points_1.csv'

COLOR_MAPPING = {114:'red', 103:'green', 98:'blue', 111:'orange', 109:'magenta', 99:'cyan', 121:'yellow', 66:'black'}

color = 66
point_list = []
i = 0


img_path = os.path.join(TEST_IMAGE_DIR, IMG_NAME)


src = cv2.imread(img_path)

def get_click_coords(event, x, y, flags, param):
    global mouse_location
    global i

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_location = (x,y)
        print(mouse_location, color)
        point_list.append([i, color, mouse_location[0], mouse_location[1]])
        i += 1

    # elif event == cv2.EVENT_LBUTTONUP:
    #     mouse_location = (x,y)
    #     print('UP', mouse_location, color)

cv2.imshow("WINDOWNAME",src)

cv2.setMouseCallback("WINDOWNAME", get_click_coords)
while True:
    k = cv2.waitKey(0)
    if k%256 in COLOR_MAPPING:
        # ESC pressed
        color = k%256
        print("Changing point color to: ", color)
    elif k%255 == 27:
        print('escape pressed...')
        break

np.savetxt(os.path.join(TEST_IMAGE_DIR,SAVE_NAME), point_list)