# Nicholas Mair
# Surface Recogntion Tray (Working Title)
# Date of Creation: 2025-04-28
# File Description:
#   -> opencv-core.py -- Getting used to the opencv library
#   -> Following Tutorials from this page: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

import cv2 as cv
import numpy as np

test = 0

if (test == 0):
    # Image Operations
    img = cv.imread('../test-data/elephant.jpg')

    cv.namedWindow('image')
    cv.imshow('image',img)
    while True:
        if cv.waitKey(1) == ord('q'):
            break
