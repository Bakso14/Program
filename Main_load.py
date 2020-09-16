import cv2
import numpy as np
from math import atan2, cos, sin, sqrt, pi
def nothing(x):
    pass

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    print(angle)
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
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
    #p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    #p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    #drawAxis(img, cntr, p1, (0, 255, 0), 100)
    #drawAxis(img, cntr, p2, (255, 255, 0), 100)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    #angle = -15*(2*pi)/360
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
    #25
    src = cv2.imread('../Video_jalan/video_1_/video_1_ 001.jpg')
    src = cv2.imread('../Video_jalan/video_1_/Testjpg.jpg')
    blur = cv2.GaussianBlur(src,(9,9),0)
    
    scale_percent = 50  
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dsize = (width, height)
    frame= cv2.resize(blur,dsize)
    crop = frame[200:360,0:640]
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    warna = cv2.inRange(hsv, HSV_Low, HSV_High)
    #res = cv2.bitwise_and(frame,frame, mask = mask)
    #erosion = cv2.erode(warna,kernel,iterations = 1)
    #dilation = cv2.dilate(mask,kernel,iterations = 1)
    #mask = cv2.morphologyEx(warna, cv2.MORPH_OPEN, kernel)
    
    contours = cv2.findContours(warna, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0] #connected component data
    #spatio temporal
    CC_O = np.zeros((len(contours)),np.int8) #connected component orientation
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        
        # Memilih luas kontur
        if area <4e1 or area > 100e2 :
            continue
        
        # Draw each contour only for visualisation purposes
        cv2.drawContours(frame, contours, i, (0, 0, 255), 1)        
        
        #cnt = contours[i]
        #rect = cv2.minAreaRect(cnt)
        #box = cv2.boxPoints(rect)
        #box = np.int0(box)
        #cv2.drawContours(crop,[box],0,(255,0,0),2)
        
        #rows,cols = frame.shape[:2]
        #[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        #lefty = int((-x*vy/vx) + y)
        #righty = int(((cols-x)*vy/vx)+y)
        #cv2.line(crop,(cols-1,righty),(0,lefty),(0,255,0),2)
        
        a = getOrientation(c, frame)
        a_derajat = 360*a/(2*pi)
        print(a_derajat)

        #spatio tempporal
        CC_O[i] = a_derajat
        
        
    
    #cv2.imshow('mask',erosion)
    cv2.imshow('hasil_Warna',warna)
    cv2.imshow('crop',frame)
    #cv2.imshow('res',res)
    #cv2.imshow('erot',erosion)
    #cv2.imshow('dilet',dilation)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cv2.destroyAllWindows()
#cap.release()