import cv2
import numpy as np
import pytesseract
import pafy
import concurrent.futures

url= 'https://www.youtube.com/watch?v=Li3FPoa3iYE'
num_set = set()
jobs=[]


h_min=8
h_max=21

s_min=112
s_max=205

v_min=75
v_max=215
pause=False
show_mask = True

def set_mask(_):
    global lower, upper
    lower = np.array([cv2.getTrackbarPos("Hue_min", "params"), cv2.getTrackbarPos("Sat_min", "params"),
                      cv2.getTrackbarPos("Val_min", "params")])
    upper = np.array([cv2.getTrackbarPos("Hue_max", "params"), cv2.getTrackbarPos("Sat_max", "params"),
                      cv2.getTrackbarPos("Val_max", "params")])
    return ()


lp_max=1000
lp_min=400
counter=0
tilt=1
starting_no=590

video = pafy.new(url)
best = video.getbest(preftype="mp4")

capture = cv2.VideoCapture()
capture.open(best.url)
s, img = capture.read()
# x_img_min = 300
# x_img_max = -1000
# y_img_min = 700
# y_img_max = -100

x_img_min = 400
x_img_max = -950
y_img_min = 800
y_img_max = -1


roi = cv2.selectROI(img)
cv2.destroyAllWindows()

cv2.namedWindow("params")
cv2.resizeWindow("params",400,200)
cv2.createTrackbar("Hue_min","params",h_min,178,set_mask)
cv2.createTrackbar("Hue_max","params",h_max,178,set_mask)
cv2.createTrackbar("Sat_min","params",s_min,255,set_mask)
cv2.createTrackbar("Sat_max","params",s_max,255,set_mask)
cv2.createTrackbar("Val_min","params",v_min,255,set_mask)
cv2.createTrackbar("Val_max","params",v_max,255,set_mask)

lower=np.array([cv2.getTrackbarPos("Hue_min","params"),cv2.getTrackbarPos("Sat_min","params"),cv2.getTrackbarPos("Val_min","params")])
upper=np.array([cv2.getTrackbarPos("Hue_max","params"),cv2.getTrackbarPos("Sat_max","params"),cv2.getTrackbarPos("Val_max","params")])


img = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
min_diff = roi[2]*roi[3]/5 #we ignore if frame is changed in less pixels than min_diff
#img = img[800:, 200:-800]
#img = img[y_img_min:y_img_max, x_img_min:x_img_max]
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
old_img_gray=np.copy(img_gray)

def write_lp(text,number):
    global num_set
    print('checking')
    if text in num_set: return False
    cv2.imwrite(text+".png",number)
    print ('saving')
    return True



def recognize(img):
    text = pytesseract.image_to_string(img,
                                      config='--user-patterns pattern.txt --oem 3 --psm 13 -c tessedit_char_whitelist=-BDFGHJKLNPRSTUVXYZ0123456789  load_system_dawg=false load_freq_dawg=false')
    return text


processor = concurrent.futures.ProcessPoolExecutor()
threader = concurrent.futures.ThreadPoolExecutor()

while True:
    if pause==False:
        old_img_gray=np.copy(img_gray)
        s, img = capture.read()
        img = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        k = np.sum(img_gray != old_img_gray)
        if k < min_diff:
            #cv2.putText(img_gray,"skipping", (20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
            #print('skipping, k=',k)
            #cv2.imshow('Contours',img_gray)
            continue
        #cv2.putText(img, 'running', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))




    if s!=True:
        continue


    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    img_blur=cv2.blur(img_hsv,(10,10),2)
    img_eroded=cv2.erode(img_blur,(30,30),iterations=2)
    mask = cv2.inRange(img_eroded, lower, upper)

    if show_mask: cv2.imshow('mask', mask)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    img_copy = np.copy(img)
    for cont in contours:
        area=cv2.contourArea(cont)


        if lp_max > area and area > lp_min:
            box=cv2.approxPolyDP(cont,1,1)
            x,y,w,h=cv2.boundingRect(box)

            cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255,0,0),3)


            number=img[y-2:y+h+2,x-3:x+w]

            if (h * w < 100):
                continue

            if 3 < (w / h) < 6:

                try:
                    #cv2.imshow('snap',number)
                    cv2.imshow('params', number)
                except:
                    print (number.shape)
                    continue
                if tilt != 0:
                    pts1 = np.float32([[0, 0+tilt],[number.shape[0], 0], [0,number.shape[1]], [number.shape[0],number.shape[1]-tilt]])
                    pts2 = np.float32([[0, 0],[number.shape[0],0], [0,number.shape[1]], [number.shape[0],number.shape[1]]])

                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    number = cv2.warpPerspective(number, M, (number.shape[1], number.shape[0]) )
                jobs.append(processor.submit (recognize,number))


                for job in jobs:
                    text=job.result()
                    if text:
                        text=text.strip()
                        if len(text)>5: print ('Read: ',text, 'len:' , len(text))
                        if len(text) != 6: continue
                        threader.submit (write_lp,text,number)
                        jobs.remove(job)


                counter+=1
                #if pause==False:
                #    for i in range(20):
                #        _, _ = capture.read()




    cv2.imshow("Contours", img_copy)


    key= cv2.waitKey(1) & 0xFF
    if key == ord ('s'):
        cv2.imwrite("car"+str(counter)+".png",img)
    elif key == ord ('q'):
        break
    elif key == ord('m'):
        if show_mask:
            cv2.destroyWindow('mask')
        show_mask = not show_mask
    elif key == ord ('p'):
        pause = not pause



capture.release()
cv2.destroyAllWindows()