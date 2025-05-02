# Nicholas Mair
# Surface Recogntion Tray (Working Title)
# Date of Creation: 2025-04-28
# File Description:
#   -> opencv-core.py -- Getting used to the opencv library
#   -> Following Tutorials from this page: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

import cv2 as cv
import numpy as np

def getCoords(event,x,y,flags,param):
    if (event == cv.EVENT_LBUTTONDOWN):
        print("X pos: ",x)
        print("Y pos: ",y)

test = 0

if (test == 0):
    # Image Operations
    img = cv.imread('../test-data/elephant.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"

    # Get Eyes ROI
    eyes = img[190:250, 300:400]
    img[100:160, 530:630] = eyes

    img[:,:,0:2] = 0

    # Show Image
    cv.namedWindow('image')
    cv.setMouseCallback('image',getCoords)
    cv.imshow('image',img)
    while True:
        if (cv.waitKey(20)==ord('q')):
            break
