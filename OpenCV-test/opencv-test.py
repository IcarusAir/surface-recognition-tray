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

test = 1


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
    cap = cv.VideoCapture()
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
        cv.imShow('live',gscale)
        if cv.waitKey(1) == ord('q'): #Display for 1 ms before running the next frame
            break #Quit on "q"
    
    # Release Captures when everything is done
    cap.release()
    cv.destroyAllWindows()