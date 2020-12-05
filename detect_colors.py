import cv2
import numpy as np
############################################
# This code opens sample image and a panel with trackbars to find the best HSV mask.  
############################################

img=cv2.imread('car.png')
h_off=50 #Define offset for every parameter for bracketing
s_off=70
v_off=50


############################################

def empty(_):
    pass

cv2.imshow('Car',img)
hsv_image=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow('Car',hsv_image)
cv2.namedWindow("params")
cv2.resizeWindow("params",400,200)
cv2.createTrackbar("Hue","params",17,179-h_off,empty)
cv2.createTrackbar("Sat","params",152,255-s_off,empty)
cv2.createTrackbar("Val","params",205,255-v_off,empty)


while True:
    h = cv2.getTrackbarPos("Hue","params")
    s = cv2.getTrackbarPos("Sat","params")
    v = cv2.getTrackbarPos("Val","params")
    lower=np.array([h-h_off,s-s_off,v-v_off])
    upper=np.array([h+h_off,s+s_off,v+v_off])
    mask=cv2.inRange(hsv_image,lower,upper)

    cv2.imshow('Mask', mask)
    


    #cv2.waitKey(1)
