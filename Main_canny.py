import cv2
import numpy as np
import time
from math import atan2, cos, sin, pi
from numpy import diff,sign
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
    #cv2.line(img,cntr,(int(x2),int(y2)),(0,255,0),1,cv2.LINE_AA)
    return angle


cannyH = 'canny high'
cannyL = 'canny low'
co = 'koreksi orientasi'
th_sse = 'threshold SSE'
img = np.zeros((25,512,3), np.uint8)
cv2.namedWindow('thresh')
cv2.createTrackbar(cannyH, 'thresh',227,255,nothing)
cv2.createTrackbar(cannyL, 'thresh',255,255,nothing)
cv2.createTrackbar(co, 'thresh',20,360,nothing)
cv2.createTrackbar(th_sse, 'thresh',75,100,nothing)

#cap = cv2.VideoCapture('../Video_jalan/video_1_.mp4')
cap = cv2.VideoCapture('../Video_jalan/Autonomous Car/Meer Selatan.mp4')
#cap = cv2.VideoCapture(0)

#menyimpan video
number_frame = 30.0 #higher frames better quality of the video
video_size = (640,160)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('hasil.mp4',fourcc, number_frame,video_size)

kernel = np.ones((3,3),np.uint8)
kernel_er = np.ones((3,3),np.uint8)

final_clstr = []
numero_clstr = []
jumlah_frame = 0
while(1):
    start = time.time()
    
    
    
    cv2.imshow('thresh',img)
    high=cv2.getTrackbarPos(cannyH, 'thresh')
    low=cv2.getTrackbarPos(cannyL, 'thresh')
    
    #src = cv2.imread('../Video_jalan/video_1_/video_1_ 001.jpg')
    #src = cv2.imread('../Video_jalan/Meer Selatan/Meer Selatan 10.jpg')
    #src = cv2.imread('../Video_jalan/Masjid/Taman Alumni, Barat Masjid 01.jpg')
    ret,src = cap.read()
    #src = cv2.imread('../Video_jalan/video_1_/Testjpg.jpg')
    #src = cv2.imread('../Video_jalan/video_1_/Test.jpg')
    if ret:
        jumlah_frame = jumlah_frame + 1
        blur = cv2.GaussianBlur(src,(9,9),0)   
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        
        scale_percent = 50  
        width = int(src.shape[1] * scale_percent / 100)
        height = int(src.shape[0] * scale_percent / 100)
        dsize = (width, height)
        
        frame= cv2.resize(blur,dsize)
        gray = cv2.resize(gray,dsize)
        
        crop = frame[200:360,0:640]
        crop_gray = gray[200:360,0:640]
        
        edges = cv2.Canny(crop_gray,high,low,apertureSize = 3,)
        
        contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0] #connected component data
        #spatio temporal
        cluster_orientation = np.zeros((2,len(contours)),np.int16) #connected component orientation
        v = 0
        for i, c in enumerate(contours):
            #panjang = len(contours[i])
            #if panjang <100 or panjang>1000:
            #    continue
            
            # Calculate the area of each contour
            area = cv2.contourArea(c)
            # Memilih luas kontur
            #if area <2e1 or area > 1e2:
             #    continue 
        
            cnt = contours[i]
            rect = cv2.minAreaRect(cnt)
            areakotak = rect[1][0]*rect[1][1]
            k = 10
            if ((rect[1][0] < k) and (rect[1][1] < k )) or ( not(rect[1][0] < k) and not(rect[1][1] < k ) ):
                continue
            #if areakotak<100 or areakotak > 2000:
            #     continue
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            #cv2.drawContours(crop,[box],0,(255,255,0),2)
            #cv2.drawContours(crop, contours, i, (0, 0, 255), 2)     
            
            a = getOrientation(c, crop)
            a_derajat = 360*a/(2*pi)
            #spatio tempporal
            cluster_orientation[0,v] = a_derajat
            cluster_orientation[1,v] = i
            v = v+1
            
            i_str = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            org = (box[3][0],box[3][1])
            fontScale = 0.5
            color = (0, 0, 0) 
            thickness = 1
            #cv2.putText(crop, i_str , org, font, fontScale, color, thickness, cv2.LINE_AA) 
            
            
            
        #Pengklasteran berdasarkan orientasi dan SSE
        koreksi_o = cv2.getTrackbarPos(co, 'thresh')
        threshold_SSE = cv2.getTrackbarPos(th_sse, 'thresh')*1000
        ayik = []
        R = 10 # maksimal dalam clstr
        N = 4 # maksimal frame cc dalam clstr
        
        a = 0
        while a < len(final_clstr):
            b = 1
            while b < len(final_clstr[a]):
                if (jumlah_frame-numero_clstr[a][b]) > N:
                    del final_clstr[a][b]
                    del numero_clstr[a][b]
                    b = b - 1
                b = b + 1
            
            if len(final_clstr[a]) < 2:
                del final_clstr[a]
                del numero_clstr[a]
                a = a - 1
            a = a + 1
                
                
        print(len(final_clstr))
        if len(final_clstr) < 1 :
            final_clstr.append([cluster_orientation[0,0]])
            numero_clstr.append([1])
        tanda = 0
        for connected_component in range(v):
            for a in range(len(final_clstr)):
                if cluster_orientation[0,connected_component] < (final_clstr[a][0] + koreksi_o) and cluster_orientation[0,connected_component] > (final_clstr[a][0] - koreksi_o):
                    final_clstr[a].append(contours[cluster_orientation[1,connected_component]])
                    numero_clstr[a].append(jumlah_frame)
                    if len(final_clstr[a]) > 1 and len(final_clstr[a]) < R:
                        gabung = np.concatenate((final_clstr[a][1:len(final_clstr[a])]), axis=0)
                        z = np.polyfit(gabung[:,0,1],gabung[:,0,0],3 ,full = True)
                        #print(z[1])
                        if z[1] < threshold_SSE:
                            tanda = 0
                            final_clstr[a][0] = (final_clstr[a][0] * (len(final_clstr[a])-1)+cluster_orientation[0,connected_component])/len(final_clstr[a])
                            break
                        else:
                            ayik.append(final_clstr[a])
                            del final_clstr[a][len(final_clstr[a])-1]
                            del numero_clstr[a][len(final_clstr[a])-1]
                            tanda = 1
                    else:
                        del final_clstr[a][len(final_clstr[a])-1]
                        del numero_clstr[a][len(final_clstr[a])-1]
                        tanda = 1
                else:
                   tanda = 1
            if tanda == 1 :
                final_clstr.append([cluster_orientation[0,connected_component]])
                numero_clstr.append([1])
                final_clstr[len(final_clstr)-1].append(contours[cluster_orientation[1,connected_component]])
                numero_clstr[len(final_clstr)-1].append(jumlah_frame)
                tanda = 0
        
        
        #Menampilkan garis hasil klaster
        ins = 0
        while ins < len(final_clstr):
            if len(final_clstr[ins]) > 1:
                gabung = np.concatenate((final_clstr[ins][1:len(final_clstr[ins])]), axis=0)
                z = np.polyfit(gabung[:,0,1],gabung[:,0,0],3 ,full = True)
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
                
                local_minmax = diff(sign(diff(y_a[:,0]))).nonzero()[0] + 1 # local min+max
    
                #print(ins,"Hasil",len(local_minmax))
                if len(local_minmax) <= 1:
                    image = cv2.polylines(crop, [garis], isClosed, color, thickness) 
                else:
                    del final_clstr[ins]
                    del numero_clstr[ins]
                    ins = ins - 1
                    
                #i_str = str(ins)
                #font = cv2.FONT_HERSHEY_SIMPLEX 
                #org = (y_a[:,0][len(y_a)-1],x_a[:,0][len(y_a)-1])
                #fontScale = 0.5
                #color = (0, 0, 0) 
                #cv2.putText(crop, i_str , org, font, fontScale, color, 1, cv2.LINE_AA)
            else:
                del final_clstr[ins]
                del numero_clstr[ins]
                ins = ins - 1
                
            ins = ins+1
        
    

                
                
        #cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        #cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
        #cv2.imshow('hasil_Warna',warna)
        cv2.imshow('crop',crop)
        #cv2.imshow('Canny',edges)
        #cv2.imshow('asli',src)
        #cv2.imshow('diation',dilation)
        #cv2.imshow('erosion',erosion)
        #cv2.imshow('gray',crop_gray)
         
        #menyimpan video
        out.write(crop)
        
        stop = time.time()
        seconds = stop - start
        fps = 1 / seconds
        print("Estimated frames per second : {0}".format(fps))
        
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
