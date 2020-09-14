import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi
def nothing(x):
    pass
    
def getOrientation(pts, img):
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
        
    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    length = 100
    x2 = cntr[0] + length*cos(angle)
    y2 = cntr[1] + length*sin(angle)
    cv2.line(img,cntr,(int(x2),int(y2)),(255,0,0),1,cv2.LINE_AA)
    return angle


img = np.zeros((300,512,3), np.uint8)

#cap = cv2.VideoCapture(0)
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
    warna = cv2.inRange(hsv, HSV_Low, HSV_High)
    erosion = cv2.erode(warna,kernel,iterations = 1)
    
    contours = cv2.findContours(warna, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        
        # Memilih luas kontur
        if area <1e1 or area > 5e2 :
            continue
        
        # Draw each contour only for visualisation purposes
        cv2.drawContours(crop, contours, i, (0, 0, 255), 2)        
        
        cnt = contours[i]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(crop,[box],0,(255,0,0),2)
        
        a = getOrientation(c, crop)
        print('a =',a)
    
    #cv2.imshow('mask',erosion)
    cv2.imshow('hasil_Warna',warna)
    cv2.imshow('crop',crop)
    #cv2.imshow('res',res)
    #cv2.imshow('erot',erosion)
    #cv2.imshow('dilet',dilation)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
#cap.release()