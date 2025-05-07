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

def updateBlendedAlpha(x):
    global img_blend
    global alpha
    global beta
    alpha = float(x)/10.0
    beta = 1 - alpha
    img_blend = cv.addWeighted(img,alpha,img2,beta,gamma)
    cv.imshow('image',img_blend)

def updateBlendedGamma(x):
    global img_blend
    global gamma
    gamma = x - 50
    img_blend = cv.addWeighted(img,alpha,img2,beta,gamma)
    cv.imshow('image',img_blend)

test = 3

if (test == 0):
    # Image Operations
    img = cv.imread('../test-data/elephant.png')
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

elif (test == 1):
    # Image Addition, Blending
    # Image Blending g(x) = alpha*f_0(x) + beta*f_1(x) + gamma
    #   alpha and beta different weights and gamma is a constant
    img = cv.imread('../test-data/elephant.jpg')
    img2 = cv.imread('../test-data/dreamy.jpg')
    img_add = cv.add(img,-50)
    alpha = 0
    beta = 1 - alpha
    gamma = -50
    img_blend = cv.addWeighted(img,alpha,img2,beta,gamma)

    # Show Image
    cv.namedWindow('image')
    cv.setMouseCallback('image',getCoords)
    cv.createTrackbar('alpha','image',0,10,updateBlendedAlpha)
    cv.createTrackbar('gamma','image',0,100,updateBlendedGamma)
    cv.imshow('image',img_blend)
    while True:
        if (cv.waitKey(20)==ord('q')):
            break

elif (test == 2):
    # Mask Creation and Bitwise Operations
    # Defining and working with non-rectangular ROIs
    img1 = cv.imread('../test-data/elephant.png')
    img2 = cv.imread('../test-data/opencv-logo-small.png')
    assert img1 is not None, "file could not be read."
    assert img2 is not None, "file could not be read."

    # Put logo in the top left corner, define ROI
    rows,cols,channels = img2.shape # nice trick, keep that in mind
    roi = img1[0:rows, 0:cols]
    # This ROI is a rectangle, we need a mask to apply to this area

    # Making our own image mask algorithm! cool!
    # First create a mask (A mask has all transparent pixels = 0 and non-transparent pixels = 1)
    gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(gray,10,255,cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)

    # Black out section of the image with inverse mask
    img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image
    img2_fg = cv.bitwise_and(img2,img2,mask = mask)

    # Add image to mask
    dst = cv.add(img1_bg,img2_fg)
    img1[0:rows,0:cols] = dst

    # Show Image
    cv.namedWindow('image')
    cv.setMouseCallback('image',getCoords)
    cv.imshow('image',img1)
    while True:
        if (cv.waitKey(20)==ord('q')):
            break

elif (test == 3):
    # Measuring Performance Functions
    cp1 = cv.getTickCount()

    # Code from Test 2 - Measure Performance
    if True:
        # Mask Creation and Bitwise Operations
        # Defining and working with non-rectangular ROIs
        img1 = cv.imread('../test-data/elephant.png')
        img2 = cv.imread('../test-data/opencv-logo-small.png')
        assert img1 is not None, "file could not be read."
        assert img2 is not None, "file could not be read."

        # Put logo in the top left corner, define ROI
        rows,cols,channels = img2.shape # nice trick, keep that in mind
        roi = img1[0:rows, 0:cols]
        # This ROI is a rectangle, we need a mask to apply to this area

        # Making our own image mask algorithm! cool!
        # First create a mask (A mask has all transparent pixels = 0 and non-transparent pixels = 1)
        gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
        ret, mask = cv.threshold(gray,10,255,cv.THRESH_BINARY)
        mask_inv = cv.bitwise_not(mask)

        # Black out section of the image with inverse mask
        img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
        # Take only region of logo from logo image
        img2_fg = cv.bitwise_and(img2,img2,mask = mask)

        # Add image to mask
        dst = cv.add(img1_bg,img2_fg)
        img1[0:rows,0:cols] = dst
    
    # Computation Time Checkpoint:
    cp2 = cv.getTickCount()
    time = (cp2 - cp1)/cv.getTickFrequency()
    print("Computation Time: ",time,"second(s).")

    # Show Image
    cv.namedWindow('image')
    cv.setMouseCallback('image',getCoords)
    cv.imshow('image',img1)
    while True:
        if (cv.waitKey(20)==ord('q')):
            break