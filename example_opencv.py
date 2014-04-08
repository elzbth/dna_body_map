#!/usr/bin/python

import cv2
import numpy as np
import sys

# create video capture
cap = cv2.VideoCapture(1)

print "to take background picture press 'b'"

while(1):
    _,background = cap.read()

    # background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    background = cv2.blur(background,(10,10))
    cv2.imshow('bck', background)
    if cv2.waitKey(33) == 27:
        cv2.destroyAllWindows()
        cap.release()
        sys.exit(0)

# key press b takes background image
    if cv2.waitKey(33) == 32:
        break





while(1):



    # read the frames
    _,frame = cap.read()

    

    # convert to hsv and find range of colors
    # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # print hsv
    # lower_blue = np.array([110,50,50])
    # upper_blue = np.array([130,255,255])
    # print upper_blue
    # thresh = cv2.inRange(hsv,lower_blue, upper_blue)
    # thresh2 = thresh.copy()

    # bw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bw = cv2.blur(bw, (10,10))
    bw = bw - background
    


    # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(bw,(5,5),0)
    # ret3,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    bw2 = bw.copy()
    ret,thresh = cv2.threshold(bw2,100,255,cv2.THRESH_BINARY_INV)
    # thresh = cv2.inRange(bw2, np.array((100)), np.array((200)))

    thresh2 = thresh.copy()

    # find contours in the threshold image
    contours,hierarchy = cv2.findContours(thresh2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # # finding contour with maximum area and store it as best_cnt
    # max_area = 0
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area > max_area:
    #         max_area = area
    #         best_cnt = cnt

    # # finding centroids of best_cnt and draw a circle there
    # M = cv2.moments(best_cnt)
    # cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    # cv2.circle(frame,(cx,cy),5,255,-1)

    sorted_contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # draw circle at centroid of all contours
    for cnt in sorted_contours[:3]:
    	M = cv2.moments(cnt)
    	if M['m00'] > 0 and M['m00'] > 0:
    		cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    		cv2.circle(frame,(cx,cy),5,255,-1)

    # Show it, if key pressed is 'Esc', exit the loop
    cv2.imshow('bw',bw)

    cv2.imshow('frame',frame)
    
    cv2.imshow('thresh',thresh)

    thresh3 = thresh.copy()
    red = [130,255,255]
    cv2.drawContours(thresh3, sorted_contours[:3], -1, red, 8)
    cv2.imshow('thresh3',thresh3)
    if cv2.waitKey(33)== 27:
        cv2.destroyAllWindows()
        cap.release()
        sys.exit(0)

