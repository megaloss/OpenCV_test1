import cv2
import pafy
import numpy as np
import pytesseract
import time

url = "https://www.youtube.com/watch?v=Li3FPoa3iYE"
#Here we define a mask found by detect_colours.py
lower=np.array([0,100,130]) #min HSV
upper=np.array([60,180,250]) #max HSV
#define an area range of license plate's contour
lp_max=550
lp_min=480
counter=0
#tes_config = r'''tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 13'''
#tes_congis = r'--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

#opening youtube stream
video = pafy.new(url)
best = video.getbest(preftype="mp4")

capture = cv2.VideoCapture()
capture.open(best.url)
#cap=cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)
capture.set(10,10)

success, img = capture.read()
print("Image size: ",img.shape)
while True:
    _, img = capture.read()
    img=img[900:1080,500:800] #this is where the cars pass by in a certain direction
    #cv2.imshow("Video",img[900:1080,500:800])
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_blur=cv2.blur(img_hsv,(10,10),2)
    
    #applying mask and detecting contours 
    mask = cv2.inRange(img_blur, lower, upper)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    for cont in contours:
        area=cv2.contourArea(cont)
        #print("area:",area)
        if lp_max > area and area > lp_min: #Choosing only contours with a certain area
            box=cv2.approxPolyDP(cont,1,1)
            x,y,w,h=cv2.boundingRect(box)
            #cv2.drawContours(img, cont, -1,(255,0,0),2)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)


            number=img[y-2:y+h+2,x-2:x+w+2] # making a snapshot of a detected contour
            print(number.shape)
            #print(img_snap.shape[1])
            if number.shape[0]>10 and number.shape[1]>40: #Checking if it has a proper shape of a typical license plate  
                num_bw=cv2.cvtColor(number,cv2.COLOR_BGR2GRAY)
                cv2.imshow('snap',num_bw)
                cv2.imwrite("lp"+str(counter)+".png",num_bw)
                counter+=1

                #trying ro recognize the number
                text=pytesseract.image_to_string(num_bw, lang ='eng', config ='--oem 3 --psm 6 -c tessedit_char_whitelist=-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                if len(text)>5: #number usually consists from 6 char/digits
                    print(text)
                for i in range (10):
                    _, img = capture.read()




    #cv2.imshow("Video", img)
    #cv2.imshow("Mask", mask)


    #saving snapshot of a license plate to file
    if cv2.waitKey(5) & 0xFF == ord ('s'):
        cv2.imwrite("car"+str(counter)+".png",img)
