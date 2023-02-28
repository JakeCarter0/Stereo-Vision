import cv2
import os



DEVICE_NAME = 'webcam_A_480p'
CAMERA_RESOLUTION = (1280,720)
SAVE_DIR = os.path.join('.\\data\\calibration images', DEVICE_NAME)

# cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam = cv2.VideoCapture(1)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

cv2.namedWindow("test")

img_counter = 10

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "{}_sample_{}.png".format(DEVICE_NAME,img_counter)
        cv2.imwrite(os.path.join(SAVE_DIR,img_name), frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()