import cv2
import numpy as np
def nothing(x):
    pass

img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

#easy assigments
hh='Hue High'
hl='Hue Low'
sh='Saturation High'
sl='Saturation Low'
vh='Value High'
vl='Value Low'

#Begin Creating trackbars for each
cv2.createTrackbar(hl, 'image',0,180,nothing)
cv2.createTrackbar(hh, 'image',180,180,nothing)
cv2.createTrackbar(sl, 'image',0,255,nothing)
cv2.createTrackbar(sh, 'image',255,255,nothing)
cv2.createTrackbar(vl, 'image',0,255,nothing)
cv2.createTrackbar(vh, 'image',255,255,nothing)

#cap = cv2.VideoCapture('../Video_jalan/video_1_.mp4')
kernel = np.ones((5,5),np.uint8)
while(1):
    cv2.imshow('image',img)
    
    hul=cv2.getTrackbarPos(hl, 'image')
    huh=cv2.getTrackbarPos(hh, 'image')
    sal=cv2.getTrackbarPos(sl, 'image')
    sah=cv2.getTrackbarPos(sh, 'image')
    val=cv2.getTrackbarPos(vl, 'image')
    vah=cv2.getTrackbarPos(vh, 'image')
        
    #_,src = cap.read()
    #src = cv2.imread('../Video_jalan/video_1_/video_1_ 001.jpg')
    src = cv2.imread('../Video_jalan/Meer Selatan/Meer Selatan 10.jpg')
    srca = cv2.imread('../Video_jalan/Meer Selatan/Meer Selatan 10.jpg',0)
    blur = cv2.GaussianBlur(src,(11,11),0)
    ret,th1 = cv2.threshold(srca,127,255,cv2.THRESH_BINARY)
    #scaling gambar
    scale_percent = 50  
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    dsize = (width, height)
    frame= cv2.resize(blur,dsize)
    crop = frame[200:360,0:640]
    dst = cv2.Canny(crop, 50, 100)
    
    #convert RGB ke HSV
    hsv = cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    HSV_Low = np.array([hul,sal,val])
    HSV_High = np.array([huh,sah,vah])
    
    #Simpan dan Load data nilai HSV
    if cv2.waitKey(1) & 0xFF == ord('a'):
        np.savetxt('data_hsv.dat', [HSV_Low, HSV_High])
        print('SAVED')
    
    if cv2.waitKey(1) & 0xFF == ord('w'):
        data = np.loadtxt('data_hsv.dat')
        HSV_Low = data[0,:]
        HSV_High = data[1,:]
        print('LOADED')
        

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
    cv2.imshow('threshold',th1)
    #cv2.imshow('hasil canny',dst)
    #cv2.imshow('res',res)
    #cv2.imshow('erot',erosion)
    #cv2.imshow('dilet',dilation)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
#cap.release()