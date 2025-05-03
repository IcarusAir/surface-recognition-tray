# Nicholas Mair
# Surface Recogntion Tray (Working Title)
# Date of Creation: 2025-04-28
# File Description:
#   -> opencv-imageprocessing.py -- Getting used to the opencv library - image processing functions
#   -> Following Tutorials from this page: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

import cv2 as cv
import numpy as np

test = 0

if (test == 0):
    # Colour Space Conversion
    # BGR to HSV

    # Set up video capture
    cap = cv.VideoCapture(0)

    # Investigate HSV values
    color = np.uint8([[[0,0,255]]])
    hsv_test = cv.cvtColor(color,cv.COLOR_BGR2HSV)
    print(hsv_test)

    # Video Capture Loop
    while True:
        # Extract frame and convert to HSV
        _,frame = cap.read()
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Making use of OpenCV inrange function, define upper and lower ranges of HSV values
        blue_lower = np.array([110,50,50]) #Around 30% of max Saturation and Value
        blue_upper = np.array([130,255,255]) #Max Saturation and Value
        mask_blue = cv.inRange(frame_hsv,blue_lower,blue_upper) #Return the given frame with white pixels in the given range and black pixels outside it

        green_lower = np.array([50,50,50]) 
        green_upper = np.array([70,255,255]) 
        mask_green = cv.inRange(frame_hsv,green_lower,green_upper)

        red_lower = np.array([10,50,50]) 
        red_upper = np.array([30,255,255]) 
        mask_red = cv.inRange(frame_hsv,red_lower,red_upper)

        # extract color from original frame (and convert back to bgr)
        frame_blue = cv.bitwise_and(frame,frame,mask=mask_blue)
        frame_green = cv.bitwise_and(frame,frame,mask=mask_green)
        frame_red = cv.bitwise_and(frame,frame,mask=mask_red)

        # # Show rest of frame in grayscale
        mask_inv = cv.bitwise_not(mask_blue)
        frame_inv = cv.bitwise_and(frame,frame,mask=mask_inv)

        # Show Image and color masks
        cv.imshow('frame',frame_inv)
        cv.imshow('mask_b',frame_blue)
        # cv.imshow('mask_g',frame_green)
        # cv.imshow('mask_r',frame_red)
        if (cv.waitKey(20)==ord('q')):
            break

    cv.destroyAllWindows()
        
        



    

    # Extract Red objects
    