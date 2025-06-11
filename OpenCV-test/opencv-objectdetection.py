# Nicholas Mair
# Surface Recogntion Tray (Working Title)
# Date of Creation: 2025-04-28
# File Description:
#   -> opencv-objectdetection.py -- Getting used to the opencv library - edge and object detection
#   -> Following Tutorials from this page: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

import cv2 as cv
import numpy as np
from matplotlib import pyplot as pltq

test = 1

def nothing(x):
    pass

if (test == 0):
    # Edge Detection
    img = cv.imread('../test-data/elephant.jpg', cv.IMREAD_GRAYSCALE)
    
    # In this function, need to output to a higher data type to avoid truncation of negative slopes.
    edges1 = cv.Laplacian(img,cv.CV_64F,ksize = 1) 
    edges2 = cv.Laplacian(img,cv.CV_64F,ksize = 3) 
    edges3 = cv.Laplacian(img,cv.CV_64F,ksize = 5) 

    cv.imshow('src',img)
    cv.imshow('dst1',edges1)
    cv.imshow('dst2',edges2)
    cv.imshow('dst3',edges3)

    while True:
        if (cv.waitKey() == ord('q')):
            break

if (test == 1):
    # Edge Detection
    cap = cv.VideoCapture(0)
    cv.namedWindow('dst')
    cv.createTrackbar('ksize','dst',1,31,nothing)

    while True:
        _,img = cap.read()
        gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
        # In this function, need to output to a higher data type to avoid truncation of negative slopes.
        kernel_size = cv.getTrackbarPos('ksize','dst')
        if (kernel_size%2 == 0):
            kernel_size += 1

        # Different operations on the video stream
        operation = 'canny'
        match operation:
            case 'laplacian':
                out = cv.Laplacian(gray_img,cv.CV_64F,ksize = kernel_size) 
            case 'canny':
                canny = cv.Canny(gray_img,100,200)
                kernel = np.ones((3,3),np.uint8)
                out = cv.dilate(canny,kernel,iterations=1)
            case _:
                # No operation
                out = gray_img


        cv.imshow('dst',out)

        if (cv.waitKey(1) == ord('q')):
            break

