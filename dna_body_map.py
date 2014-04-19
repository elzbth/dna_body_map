#!/usr/bin/python

# import opencv
import cv2
import numpy as np
import sys

# import opengl stuff
from OpenGL.GL import *
from OpenGL.GLU import *
import random
from math import * # trigonometry
import Image



# empty callback function, necessary for the trackbars 
def no_func(i):
    pass

# create video capture
cap = cv2.VideoCapture(1)
cap.set(cv2.cv.CV_CAP_PROP_FPS, 60)
print "to take background picture press space"

# this background subtractor is supposedly la leche but is not recognized. Problem with CV version?
# fgbg = cv2.createBackgroundSubtractorMOG()

# loop while waiting for background image capture
while(1):
    _,background = cap.read()

    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    background = cv2.blur(background,(10,10))
    cv2.imshow('bck', background)

    # press Esc kills all
    if cv2.waitKey(33) == 27:
        cv2.destroyAllWindows()
        cap.release()
        sys.exit(0)

    # key press space takes background image and breaks 
    if cv2.waitKey(33) == 32:
        break


# make wndow for trackbars for various variables
cv2.namedWindow('vars')

# make viz windows

cv2.namedWindow('bw', cv2.WINDOW_AUTOSIZE)

cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

cv2.namedWindow('thresh', cv2.WINDOW_AUTOSIZE)

# for window that will be mapping, enable openGL
cv2.namedWindow('mapping', cv2.WINDOW_AUTOSIZE, cv2.WINDOW_OPENGL)

# cv2.namedWindow('contours', cv2.WINDOW_AUTOSIZE)



# adjustable variables
# cutoff for thresholding
cutoff = 100
cv2.createTrackbar('thresh_cutoff', 'vars', cutoff, 255, no_func)

# approx_frac = 10
# cv2.createTrackbar('approx_frac', 'vars', approx_frac, 500, no_func)

# kernel size for closing
kernel_size = 5
cv2.createTrackbar('kernel_size', 'vars', kernel_size, 500, no_func)


# loop over video frames
while(1):


    # read the frames
    _,frame = cap.read()
    
    
    # naive background substraction: subtract first frame
    bw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bw = cv2.blur(bw, (10,10))
    bw = bw - background

    # fancy background subtraction not wrking now
    # bw = frame
    # bw2 = fgbg.apply(bw)


    cv2.imshow('bw',bw)

    # operations on images modify the source image, so make a copy before
    # bw2 = bw.copy()

    cutoff = cv2.getTrackbarPos('thresh_cutoff', 'vars')
    ret,thresh = cv2.threshold(bw.copy(),cutoff,255,cv2.THRESH_BINARY)
    # thresh = cv2.inRange(bw2, np.array((100)), np.array((200)))
    cv2.imshow('thresh',thresh)


    # reduce back blobs within white profile with "closing"
    # kernel size influences what size region to close
    kernel_size = cv2.getTrackbarPos('kernel_size', 'vars')
    if kernel_size < 1:
        kernel_size = 1
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    closing = cv2.morphologyEx(thresh.copy(), cv2.MORPH_CLOSE, kernel)
    

    cv2.imshow('closing',closing)

    # find contours in the threshold image
    contours,hierarchy = cv2.findContours(closing.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


    # sort contours by decreasing area
    sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # this is to smooth contours -- not needed i think
    # approx_frac = cv2.getTrackbarPos('approx_frac', 'vars')


    # for top four contours:
    for cnt in sorted_contours[:3]:

        # calculate moments and centroid, and draw a circle at centroid
        M = cv2.moments(cnt)
        if M['m00'] > 0 and M['m00'] > 0:
            cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            cv2.circle(frame,(cx,cy),5,255,-1)


        # calculate smoothed contour
        # epsilon = (approx_frac / 10000.0) * cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt,epsilon,True)
        # cv2.drawContours(frame, [approx], -1, (0,0,255), 2)

    # calculate bouding rectangle of best contour
    x,y,w,h = cv2.boundingRect(sorted_contours[0])
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # draw best contour over frame in red
    cv2.drawContours(frame, sorted_contours[0], -1, (0,0,255), 2)

    ## make mask to mask in black all the image that is not inside the contour:
    # make matrix of zeros (black) same size as frame
    mask = np.zeros(frame.shape,np.uint8)

    # draw best contour in white on black mask, and set to fill with thickness=-1
    cv2.drawContours(mask,[sorted_contours[0]],0,(255,255,255),thickness=-1)

    #print mask

    # invert the mask, now profile is balck (0) and background is white (255)
    mask_inv = cv2.bitwise_not(mask)

    
    cv2.imshow('mask', mask)
    cv2.imshow('inv_mask', mask_inv)



    # add mask to frame, pixels inside the profile are unchanged (+0), those in the background are maxed out (+255)
    masked_frame = cv2.add(frame, mask_inv)

    # subtract mask from frame, pixels inside the mask are unchanged (-0), those in the background go to 0 (255-255)
    db_masked_frame = masked_frame - mask_inv

    #display various windows
    cv2.imshow('frame',frame)
    cv2.imshow('frame_masked',masked_frame)
    cv2.imshow('db_masked_frame', db_masked_frame)

    # if key pressed is 'Esc', exit the loop


    if cv2.waitKey(33)== 27:
        cv2.destroyAllWindows()
        cap.release()
        sys.exit(0)

