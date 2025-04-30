# Nicholas Mair
# Surface Recogntion Tray (Working Title)
# Date of Creation: 2025-04-28
# File Description:
#   -> opencv-test.py -- Getting used to the opencv library
#   -> Following Tutorials from this page: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

import cv2 as cv
import numpy as np
import scipy
import sys

test = 3

# Blank function that runs when trackbar is updated
def nothing(x):
    pass

# Create mouse event function
def draw_circle_dbl_click(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),100,(255,0,0),-1)

if (test == 0):
    img = cv.imread(cv.samples.findFile("../data/elephant.jpg")) # Need the specific dir?

    if img is None:
        sys.exit("Could not read the image") # Python system exit code

    cv.imshow("Nightmare Fuel", img) # Show the image that has been found
    press = cv.waitKey(0) # Wait a number of milliseconds for the user to hit a key (0 - forever)

    # The press variable is assigned a unicode of the key pressed, if it was s, save the image with write
    if (press == ord("s")): #ord - return the unicode value of the character
        cv.imwrite("../data/elephant.png", img) #Save to a PNG file
elif (test == 1) :
    cap = cv.VideoCapture(0) # Index of the camera or a video file
    if not cap.isOpened():
        sys.exit("Could not open camera")

    while True:
        # Initiate a frame-by-frame capture
        # Multiple cores? One captures and pipes images, one processes
        read, frame = cap.read()

        # check if next frame is read correctly
        if not read:
            print("Could not read next frame (stream ended?). Exiting.")
            break

        # Frame operations
        # Change to grayscale (Interesting, image processing for better capture)
        gscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Display the frame
        cv.imshow('live',gscale)
        if cv.waitKey(1) == ord('q'): #Display for 1 ms before running the next frame
            break #Quit on "q"
    
    # Release Captures when everything is done
    cap.release()
    cv.destroyAllWindows()
elif (test == 2):
    # Create a Blank Image
    img = np.zeros((300,512,3),np.uint8) # Array of zeros (8 bits)
    cv.namedWindow("image")

    # Create Trackbars for colour changes
    cv.createTrackbar('R','image',0,255,nothing)
    cv.createTrackbar('G','image',0,255,nothing)
    cv.createTrackbar('B','image',0,255,nothing)

    # Create switch for ON/OFF functions
    switch = '0 : OFF \n1 : ON'
    cv.createTrackbar(switch,'image',0,1,nothing)

    while(1):
        cv.imshow('image',img)
        k = cv.waitKey(1) & 0xFF
        if (k == 27) :
            break

        #Get positions of the trackbars
        r = cv.getTrackbarPos('R','image')
        g = cv.getTrackbarPos('G','image')
        b = cv.getTrackbarPos('B','image')
        s = cv.getTrackbarPos(switch,'image')

        if s==0:
            img[:] = 0
        else:
            img[:] = [b,g,r]
    
    cv.destroyAllWindows()
elif (test == 3):
    # Create a white image and a window
    img = np.ones((512,512,3),np.uint8)
    cv.namedWindow('image')
    # Bind window to mouse functions
    cv.setMouseCallback('image',draw_circle_dbl_click)

    while (1):
        cv.imshow('image',img)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()