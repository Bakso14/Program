import cv2
import numpy as np
import time
from math import atan2, cos, sin, pi
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
    #cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    length = 100
    x2 = cntr[0] + length*cos(angle)
    y2 = cntr[1] + length*sin(angle)
    #cv2.line(img,cntr,(int(x2),int(y2)),(255,0,0),1,cv2.LINE_AA)
    return angle


cap = cv2.VideoCapture('../Video_jalan/video_1_.mp4')
#cap = cv2.VideoCapture(0)

while(1):
    start = time.time()
    data = np.loadtxt('data_hsv.dat')
    HSV_Low = data[0,:]
    HSV_High = data[1,:]
           
    #src = cv2.imread('../Video_jalan/video_1_/video_1_ 001.jpg')
    _,src = cap.read()
    #src = cv2.imread('../Video_jalan/video_1_/Testjpg.jpg')
    #src = cv2.imread('../Video_jalan/video_1_/Test.jpg')
    blur = cv2.GaussianBlur(src,(9,9),0)
    
    scale_percent = 50  
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dsize = (width, height)
    frame= cv2.resize(blur,dsize)
    crop = frame[200:360,0:640]
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    warna = cv2.inRange(hsv, HSV_Low, HSV_High)
    contours = cv2.findContours(warna, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0] #connected component data
    #spatio temporal
    CC_O = np.zeros((2,len(contours)),np.int16) #connected component orientation
    v = 0
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Memilih luas kontur
        if area <4e1 or area > 10e2 :
            continue
        #cv2.drawContours(crop, contours, i, (0, 0, 255), 2)      
        a = getOrientation(c, crop)
        a_derajat = 360*a/(2*pi)
        #spatio tempporal
        CC_O[0,v] = a_derajat
        CC_O[1,v] = i
        v = v+1
        
    #Pengklasteran berdasarkan orientasi
    koreksi_o = 45
    threshold_SSE = 100000
    final_clstr = []
    final_clstr.append([CC_O[0,0]])
    tanda = 0
    for wow in range(v):
        for a in range(len(final_clstr)):
           if CC_O[0,wow] < (final_clstr[a][0] + koreksi_o) and CC_O[0,wow] > (final_clstr[a][0] - koreksi_o):
               final_clstr[a].append(contours[CC_O[1,wow]])
               if len(final_clstr[a]) > 1:
                   gabung = np.concatenate((final_clstr[a][1:len(final_clstr[a])]), axis=0)
               z = np.polyfit(gabung[:,0,1],gabung[:,0,0],1 ,full = True)
               #print(z[1])
               if z[1] < threshold_SSE:
                   tanda = 0
               else:
                   del final_clstr[a][len(final_clstr[a])-1]
                   tanda = 1
               break
           else:
              tanda = 1
        if tanda == 1 :
            final_clstr.append([CC_O[0,wow]])
            final_clstr[len(final_clstr)-1].append(contours[CC_O[1,wow]])
            tanda = 0
    
    #Menampilkan garis hasil klaster
    for ins in range(len(final_clstr)):
        if len(final_clstr[ins]) > 1:
            gabung = np.concatenate((final_clstr[ins][1:len(final_clstr[ins])]), axis=0)
        z = np.polyfit(gabung[:,0,1],gabung[:,0,0],1 ,full = True)
        p = np.poly1d(z[0])
        x_a = np.arange(min(gabung[:,0,1]),max(gabung[:,0,1]))
        x_a= x_a.reshape((-1, 1))
        y_a = p(x_a)
        y_a = y_a.astype(np.int32)
        y_a = y_a.reshape((-1, 1))
        garis = np.concatenate((y_a,x_a), axis=1)
        garis = garis.reshape((-1, 1, 2))
        color = (255, 0, 0)   
        isClosed = False
        thickness = 2
        image = cv2.polylines(crop, [garis], isClosed, color, thickness) 
    
    stop = time.time()
    seconds = stop - start
    fps = 1 / seconds
    print("Estimated frames per second : {0}".format(fps))
    cv2.imshow('hasil_Warna',warna)
    cv2.imshow('crop',crop)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()