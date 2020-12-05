import cv2
import pafy
import numpy as np
import pytesseract
import time

url = "https://www.youtube.com/watch?v=Li3FPoa3iYE"
lower=np.array([0,100,130])
upper=np.array([60,180,250])
lp_max=550
lp_min=480
counter=0
tes_config = r'''tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --oem 3 --psm 13'''
tes_congis = r'--oem 3 --psm 6 -c tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
video = pafy.new(url)
best = video.getbest(preftype="mp4")

capture = cv2.VideoCapture()
capture.open(best.url)
#cap=cv2.VideoCapture(0)
#cap.set(3,640)
#cap.set(4,480)
capture.set(10,10)
#capture.set(cv2.CV_CAP_PROP_BUFFERSIZE, 1)
success, img = capture.read()
print("Image size: ",img.shape)
while True:
    _, img = capture.read()

    #cv2.putText(img,'bad boy', (300,270),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
                  #cv2.FILLED)
    img=img[900:1080,500:800]
    #cv2.imshow("Video",img[900:1080,500:800])
    #img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_blur=cv2.blur(img_hsv,(10,10),2)

    mask = cv2.inRange(img_blur, lower, upper)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cont in contours:
        area=cv2.contourArea(cont)
        #print("area:",area)
        if lp_max > area and area > lp_min:
            box=cv2.approxPolyDP(cont,1,1)
            x,y,w,h=cv2.boundingRect(box)
            #cv2.drawContours(img, cont, -1,(255,0,0),2)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)


            number=img[y-2:y+h+2,x-2:x+w+2]
            print(number.shape)
            #print(img_snap.shape[1])
            if number.shape[0]>10 and number.shape[1]>40:
                num_bw=cv2.cvtColor(number,cv2.COLOR_BGR2GRAY)
                cv2.imshow('snap',num_bw)
                cv2.imwrite("lp"+str(counter)+".png",num_bw)
                counter+=1


                text=pytesseract.image_to_string(num_bw, lang ='eng', config ='--oem 3 --psm 6 -c tessedit_char_whitelist=-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                if len(text)>5:
                    print(text)
                for i in range (10):
                    _, img = capture.read()




    #cv2.imshow("Video", img)
    #cv2.imshow("Mask", mask)


    #cv2.waitKey(0)
    if cv2.waitKey(5) & 0xFF == ord ('s'):
        cv2.imwrite("car"+str(counter)+".png",img)
