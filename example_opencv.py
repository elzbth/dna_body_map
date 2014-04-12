#!/usr/bin/python

import cv2
import numpy as np
import sys


def no_func(i):
    pass

# create video capture
cap = cv2.VideoCapture(0)

print "to take background picture press space"

# fgbg = cv2.createBackgroundSubtractorGMG()

while(1):
    _,background = cap.read()

    background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)

    background = cv2.blur(background,(10,10))
    cv2.imshow('bck', background)
    if cv2.waitKey(33) == 27:
        cv2.destroyAllWindows()
        cap.release()
        sys.exit(0)

# key press space takes background image
    if cv2.waitKey(33) == 32:
        break


# make wndow for trackbars for various variables
cv2.namedWindow('vars')

# make viz windows

cv2.namedWindow('bw', cv2.WINDOW_AUTOSIZE)

cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)

cv2.namedWindow('thresh', cv2.WINDOW_AUTOSIZE)

# cv2.namedWindow('contours', cv2.WINDOW_AUTOSIZE)



# adjustable variables
cutoff = 100
cv2.createTrackbar('thresh_cutoff', 'vars', cutoff, 255, no_func)

approx_frac = 10
cv2.createTrackbar('approx_frac', 'vars', approx_frac, 500, no_func)

kernel_size = 5
cv2.createTrackbar('kernel_size', 'vars', kernel_size, 500, no_func)



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

    bw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bw = cv2.blur(bw, (10,10))
    bw = bw - background
    # bw2 = fgbg.apply(bw)
    cv2.imshow('bw',bw)


    bw2 = bw.copy()

    cutoff = cv2.getTrackbarPos('thresh_cutoff', 'vars')
    ret,thresh = cv2.threshold(bw2,cutoff,255,cv2.THRESH_BINARY)
    # thresh = cv2.inRange(bw2, np.array((100)), np.array((200)))
    cv2.imshow('thresh',thresh)

    # thresh2 = thresh.copy()
    kernel_size = cv2.getTrackbarPos('kernel_size', 'vars')
    if kernel_size < 1:
        kernel_size = 1
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    closing = cv2.morphologyEx(thresh.copy(), cv2.MORPH_CLOSE, kernel)
    

    #thresh3 = thresh.copy()

    # find contours in the threshold image
    contours,hierarchy = cv2.findContours(closing.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

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

    approx_frac = cv2.getTrackbarPos('approx_frac', 'vars')
    # for top four contours:
    for cnt in sorted_contours[:3]:

        # calculate moments and centroid
        M = cv2.moments(cnt)
        if M['m00'] > 0 and M['m00'] > 0:
            cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            cv2.circle(frame,(cx,cy),5,255,-1)


        # calculate approximate contour

        # epsilon = (approx_frac / 10000.0) * cv2.arcLength(cnt,True)
        # approx = cv2.approxPolyDP(cnt,epsilon,True)
        # cv2.drawContours(frame, [approx], -1, (0,0,255), 2)

    x,y,w,h = cv2.boundingRect(sorted_contours[0])
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.drawContours(frame, sorted_contours[0], -1, (0,0,255), 2)

    mask = np.zeros(frame.shape,np.uint8)
    #mask.fill(1)
    cv2.drawContours(mask,[sorted_contours[0]],0,(255,255,255),thickness=-1)
    print mask 
    mask_inv = cv2.bitwise_not(mask)

    
    cv2.imshow('mask', mask)
    cv2.imshow('inv_mask', mask_inv)


    cv2.imshow('closing',closing)

    masked_frame = cv2.add(frame, mask_inv)
    db_masked_frame = masked_frame - mask_inv

    cv2.imshow('frame',frame)
    cv2.imshow('frame_masked',masked_frame)
    cv2.imshow('db_masked_frame', db_masked_frame)

    # Show it, if key pressed is 'Esc', exit the loop
    

    
    
    


    

    if cv2.waitKey(33)== 27:
        cv2.destroyAllWindows()
        cap.release()
        sys.exit(0)

def no_func(int):
    return 0