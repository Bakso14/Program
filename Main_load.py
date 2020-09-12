import cv2
import numpy as np
def nothing(x):
    pass

img = np.zeros((300,512,3), np.uint8)

cap = cv2.VideoCapture(0)
kernel = np.ones((5,5),np.uint8)
while(1):
    
    data = np.loadtxt('data_hsv.dat')
    HSV_Low = data[0,:]
    HSV_High = data[1,:]
        
    #_,src = cap.read()
    src = cv2.imread('../Video_jalan/video_1_/video_1_ 001.jpg')
    blur = cv2.GaussianBlur(src,(9,9),0)
    
    scale_percent = 50  
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dsize = (width, height)
    frame= cv2.resize(blur,dsize)
    crop = frame[200:360,0:640]
    
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    #HSV_Low = np.array([hul,sal,val])
    #HSV_High = np.array([huh,sah,vah])
    
    
        
    warna = cv2.inRange(hsv, HSV_Low, HSV_High)
    #res = cv2.bitwise_and(frame,frame, mask = mask)
    #erosion = cv2.erode(mask,kernel,iterations = 1)
    #dilation = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.morphologyEx(warna, cv2.MORPH_OPEN, kernel)
    
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        
        # Memilih luas kontur
        if area < 1e2 or area < 1e5:
            continue
        
        # Draw each contour only for visualisation purposes
        #cv2.drawContours(frame, contours, i, (0, 0, 255), 2)        
        
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(frame,[box],0,(255,0,0),2)
        
        rows,cols = frame.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        #cv2.line(frame,(cols-1,righty),(0,lefty),(0,255,0),2)
    
    #cv2.imshow('frame',frame)
    cv2.imshow('mask',warna)
    cv2.imshow('crop',crop)
    #cv2.imshow('res',res)
    #cv2.imshow('erot',erosion)
    #cv2.imshow('dilet',dilation)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()