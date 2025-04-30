# Nicholas Mair
# Surface Recogntion Tray (Working Title)
# Date of Creation: 2025-04-28
# File Description:
#   -> paint-alike.py - applying what I learned in previous tutorials
#   -> to make a painting system.
import cv2 as cv
import numpy as np

#Define image and global variables
color = (0,0,0)
radius = 1
drawing = False
img = np.ones((512,512,3), np.uint8)

def nothing(x):
    pass

def paint_screen(event,x,y,flags,param):
    # Pressing the mouse starts a drawing
    global drawing
    if (event == cv.EVENT_LBUTTONDOWN):
        drawing = 1
    # Unpressing the mouse ends the drawing
    elif (event == cv.EVENT_LBUTTONUP):
        drawing = 0
    # While Drawing, draw a circle of set color and radius in current position
    elif (drawing and event == cv.EVENT_MOUSEMOVE):
        cv.circle(img,(x,y),radius,color,-1)

# Create window and trackbars
cv.namedWindow('image') # A Named window can have trackbars pinned to it
cv.setMouseCallback('image',paint_screen)
cv.createTrackbar('R','image',0,255,nothing) #'image' is passed as the named window to add the trackbar to.
cv.createTrackbar('G','image',0,255,nothing)
cv.createTrackbar('B','image',0,255,nothing)
cv.createTrackbar('Radius','image',1,50,nothing)

drawing = 0

while (1):
    # Show/Quit Check
    cv.imshow('image',img)
    key = cv.waitKey(3)
    if key == ord('q'):
        break
    elif key == ord('p'):
        print("Current Values:")
        print("Color: R =",r,", G =",g,", B =",b)
        print("Radius:",radius)
    
    # Get current trackbar positions
    r = cv.getTrackbarPos('R','image') #title, window
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    radius = cv.getTrackbarPos('Radius','image')

    # Update global color value
    color = (b,g,r)




