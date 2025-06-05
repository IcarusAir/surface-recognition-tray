# Nicholas Mair
# Surface Recogntion Tray (Working Title)
# Date of Creation: 2025-04-28
# File Description:
#   -> opencv-imageprocessing.py -- Getting used to the opencv library - image processing functions
#   -> Following Tutorials from this page: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

test = 5

def nothing(x):
    pass

if (test == 0):
    # Colour Space Conversion
    # BGR to HSV

    # Set up video capture
    cap = cv.VideoCapture(0)

    # # Investigate HSV values
    # color = np.uint8([[[0,0,255]]])
    # hsv_test = cv.cvtColor(color,cv.COLOR_BGR2HSV)
    # print(hsv_test)

    # Video Capture Loop
    while True:
        # Extract frame and convert to HSV
        _,frame = cap.read()
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # Making use of OpenCV inrange function, define upper and lower ranges of HSV values
        blue_lower = np.array([90,50,50]) #Around 30% of max Saturation and Value
        blue_upper = np.array([130,255,255]) #Max Saturation and Value
        mask_blue = cv.inRange(frame_hsv,blue_lower,blue_upper) #Return the given frame with white pixels in the given range and black pixels outside it

        green_lower = np.array([50,40,40]) 
        green_upper = np.array([80,255,255]) 
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
        
elif (test == 1) :
    # Geometric Transformations and Matrix Math
    img = cv.imread('../test-data/dreamy.jpg')
    cv.imshow('original',img)

    # ================ Scaling example ====================
    # Scaling can be done with resize()
    # Can either specify a size or pass None and specify the vertical and horizontal scale factor
    scale = cv.resize(img,None,fx=2,fy=0.5,interpolation=cv.INTER_LINEAR)
    cv.imshow('scaled with resize()',scale)

    # Many operations can be performed via the warpAffine function
    # This function takes the input image matrix applies a transformation

    # ================ Translation example ====================
    M_translate = np.float32([[1,0,-45],[0,1,50]]) #Translate by -45 horizontally and 50 vertically
    # Affine matrix takes type of float32, warpAffine takes the final image shape as parameters,
    # We don't want to change the image size so we pass the original shape (cols, rows) since it is width, height
    rows, cols, color = img.shape
    move = cv.warpAffine(img,M_translate,(cols,rows))
    cv.imshow('translation',move)

    # ================ Rotation example ====================
    # Did an example application of the rotation matrix:
    # [ cos(theta)  -sin(theta) ]
    # [ sin(theta)   cos(theta) ]
    # Multiplying by this matrix will rotate by theta CW
    # OpenCV operates slightly differently, we will visit this later if we need it...

    cv.waitKey()
    cv.destroyAllWindows()

elif (test == 2):
    # thresholding, important for this project
    # Import image to be thresholded
    img_val = 0
    if (img_val):
        img = cv.imread('../test-data/dreamy.jpg')
    else:
        img = cv.imread('../test-data/elephant.jpg', cv.IMREAD_GRAYSCALE)
    # Double picture dimensions
    img = cv.resize(img,None,fx=2,fy=2,interpolation=cv.INTER_LINEAR)
    cv.imshow('img',img)

    # Convert to RGB Colourspace for matplotlib
    rgb = cv.cvtColor(img, 4)
    
    # Perform all basic thresholding on the imported image
    ret,thresh1 = cv.threshold(rgb,127,255,cv.THRESH_BINARY)
    ret,thresh2 = cv.threshold(rgb,127,255,cv.THRESH_BINARY_INV)
    ret,thresh3 = cv.threshold(rgb,127,255,cv.THRESH_TRUNC)
    ret,thresh4 = cv.threshold(rgb,127,255,cv.THRESH_TOZERO)
    ret,thresh5 = cv.threshold(rgb,127,255,cv.THRESH_TOZERO_INV)

    titles = ['Original','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [rgb,thresh1,thresh2,thresh3,thresh4,thresh5]

    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    
    plt.show()

elif (test == 3):
    # Adaptive Thresholding
    img = cv.imread('../test-data/elephant.jpg',cv.IMREAD_GRAYSCALE)

    ret, thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    thresh2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,15,0)
    thresh3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,15,0)
    # The blockSize parameter refers to an n x n matrix of pixels (refered to as a neighbourhood)
    # In this case we are looking at a 5x5 area around each pixel to calculate a threshold value
    # For Gaussian, this 5x5 neighbourhood would follow the 2d discrete Gaussian function, with the central pixel being 0,0.
    
    # Show the thresholds in a plot
    images = [img,thresh1,thresh2,thresh3]
    titles = ['original','global','mean','gaussian']

    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

elif (test == 4):
    # Different types of blurring filters, let's do video this time:
    capture = cv.VideoCapture(0)
    cv.namedWindow('original')
    cv.namedWindow('box blur')
    cv.namedWindow('gaussian')

    # Create Trackbar for blur factor
    cv.createTrackbar('Blur','original',1,100,nothing) #'image' is passed as the named window to add the trackbar to.

    # Loop to capture frames:
    while (True):
        _,frame = capture.read()

        # Perform filtering
        # Gaussian 7x7 kernel.
        g_kernel = np.array([[0, 0, 1, 2, 1, 0, 0],
                          [0, 3, 13, 22, 13, 2, 0],
                          [1, 13, 59, 97, 59, 13, 1],
                          [2, 22, 97, 159, 97, 22, 2],
                          [1, 13, 59, 97, 59, 13, 1],
                          [0, 3, 13, 22, 13, 2, 0],
                          [0, 0, 1, 2, 1, 0, 0]],np.float32)/1003
        
        # Box Blur with trackbar
        blur_factor = cv.getTrackbarPos('Blur','original')
        if (blur_factor == None or blur_factor == 0):
            blur_factor = 1
        # The trackbar sets the size of the np array dimension. So 11 creates an 11x11 box blur kernel.
        box_kernel = np.ones((blur_factor,blur_factor),np.float32)/(blur_factor*blur_factor)
        
        f_frame = cv.filter2D(frame,-1,box_kernel)

        # Create a Gaussian Blur (must be a positive odd integer, this math is just collapsing the blur factor to an add value)
        if (blur_factor%2 == 0):
            g_factor = blur_factor - 1
        else:
            g_factor = blur_factor
        g_frame = cv.GaussianBlur(frame,(g_factor,g_factor),0)

        # Show frames:
        cv.imshow('original',frame)
        cv.imshow('box blur',f_frame)
        cv.imshow('gaussian',g_frame)

        if (cv.waitKey(1) == ord('q')):
            break

elif (test == 5):
    # Morphological operations
    j = cv.imread('../test-data/j.png',cv.IMREAD_GRAYSCALE)

    cv.namedWindow('original')
    cv.namedWindow('dilated')
    cv.namedWindow('gradient')
    cv.namedWindow('closed')

    # Perform operations
    kernel = np.ones((3,3),np.uint8)
    erosion = cv.erode(j,kernel,iterations = 1)
    dilation = cv.dilate(j,kernel,iterations = 1)

    # Closing a loop
    gradient = cv.morphologyEx(dilation, cv.MORPH_GRADIENT, kernel)
    closed = cv.morphologyEx(gradient, cv.MORPH_CLOSE, kernel)
    
    # Show Results
    cv.imshow('original',j)
    cv.imshow('dilated',dilation)
    cv.imshow('gradient',gradient)
    cv.imshow('closed',closed)

    cv.waitKey()

    