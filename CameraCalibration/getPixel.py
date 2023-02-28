import cv2
import matplotlib.pyplot as plt
import os

TEST_IMAGE_DIR = '.\\data\\test images'
IMG_NAME = 'webcam_A_sample_0.png'




img_path = os.path.join(TEST_IMAGE_DIR, IMG_NAME)


src = cv2.imread(img_path)

def get_click_coords(event, x, y, flags, param):
    global mouse_location

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_location = (x,y)
        print('DOWN', mouse_location)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_location = (x,y)
        print('UP', mouse_location)

cv2.imshow("WINDOWNAME",src)
cv2.setMouseCallback("WINDOWNAME", get_click_coords)
cv2.waitKey(0)