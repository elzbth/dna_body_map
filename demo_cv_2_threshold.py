#!/usr/bin/python
import freenect
import cv2
import frame_convert
import numpy as np
import cv2.cv as cv






def change_near_threshold(value):
    near_threshold = value


def change_far_threshold(value):
    far_threshold = value

def no_func(i):
    print "func %d" % (i)


cv2.namedWindow('Depth')
cv2.namedWindow('Thresh')


near_threshold = 0
far_threshold = 100 


cv2.createTrackbar('near_threshold', 'Thresh', 0, 2048,  no_func)
cv2.createTrackbar('far_threshold',     'Thresh', 0, 2048, no_func)

print('Press ESC in window to stop')


while 1:





    near_threshold = cv2.getTrackbarPos('near_threshold','Thresh')
    far_threshold = cv2.getTrackbarPos('far_threshold','Thresh')
    
    print "%d < depth < %d" % (near_threshold, far_threshold)
    depth, timestamp = freenect.sync_get_depth()

    cv2.imshow('Depth', depth)

    thresh = 255 * np.logical_and(depth > near_threshold,
                                 depth < far_threshold)
    thresh = thresh.astype(np.uint8)
    

    cv2.imshow('Thresh', thresh)





    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()

        break
