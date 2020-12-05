import cv2
import pytesseract
from pytesseract import Output
import glob
#import numpy as np
luck,count=0,0
import pandas as pd
data=pd.read_csv('myrdw.csv',index_col='Kenteken')
found_data=pd.DataFrame()


files=(glob.glob("*.png"))
print(list(files))

for file in files:
    img=cv2.imread(file)
    #cv2.imshow('License plate',img)


    text=pytesseract.image_to_string(img, lang ='eng', config ='--oem 3 --psm 13 load_system_dawg=false load_freq_dawg=false -c tessedit_char_whitelist=-ABCDEFGHIJKLMNOPRSTUVXYZ0123456789')
    #print (text)
    #text_data=pytesseract.image_to_data(img, config ='--oem 3 --psm 13 load_system_dawg=false load_freq_dawg=false -c tessedit_char_whitelist=-ABCDEFGHIJKLMNOPRSTUVXYZ0123456789', output_type=Output.DICT)
    #print(text_data['conf'], text_data['text'])
    count+=1
    text=text.replace('-','')[:6]

    print(text)
    if len(text)!=6: continue

    found=data[data.index == text]
    if len(found)>0:
        print(found)
        luck+=1
        found_data=found_data.append(found)
        print(file)


    #cv2.waitKey(0)

print("Total:", count)
print("Found:", luck)
found_data.to_csv('found.csv')