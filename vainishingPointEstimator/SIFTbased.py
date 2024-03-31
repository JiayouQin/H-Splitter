'''
Jiayou Qin 03/29/2024
SIFT vainishing point estimation
'''

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

video_path = "../../gpt/Scene_Park.mp4" #change this file name as needed
# video_path = "rotationTest.mp4"
cap = cv.VideoCapture(video_path)


visual_intensity = 1
visualize_points = False
visualize_cursor = True

for i in range(50): #skip frames, must be larger than 1
    success, frame = cap.read()
frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
sift = cv.SIFT_create()
frame = cv.resize(frame,(800,800))
y,x = frame.shape[:2]
center = np.array([x/2,y/2], dtype=float)
cursor = center.copy()

last_frame = frame
kp, des = sift.detectAndCompute(frame,None)

cv.namedWindow('frame',cv.WINDOW_NORMAL)
cv.namedWindow('mask',cv.WINDOW_NORMAL)
cv.imshow('frame', frame)
cv.waitKey(0) #press key to begin

rot_m = np.array((2,3),)
axes = np.float32([[1,0], [0,1]])*50

last_m = None
offset = np.array([100,200]) #rotation indicator
mask_accu = np.zeros(frame.shape[:2],dtype=float)
accumulator = np.zeros(frame.shape[:2],dtype=float) #due to voting scheme this data type should be large
down_sample = True

while True:
    for i in range(5):
        success, frame = cap.read()
    if not success:
        break
    frame = cv.resize(frame,(800,800))
    frame_org = frame.copy()
    angles = np.array([],dtype=float)
    delta = np.array([],dtype=float)
    p1s = np.array([],dtype=float)
    p2s = np.array([],dtype=float)
    
    frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    frame_trans_p2 = frame.copy()
    out_frame = frame.copy()
        
    kp_,des_ = kp,des
    kp, des = sift.detectAndCompute(frame,None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des,des_,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            p1 = np.array(kp[m.queryIdx].pt, dtype=float)
            p2 = np.array(kp_[m.trainIdx].pt, dtype=float)
            p1s = np.insert(p1s, 0, p1) #delta vector, indicating pixel change
            p2s = np.insert(p2s, 0, p2)
            
    p1s = p1s.reshape(-1,2)
    p2s = p2s.reshape(-1,2)
    


    trans_m,_ = cv.estimateAffine2D(p1s,p2s)
    axes = np.matmul(axes,trans_m[:,:2])

    # apply affine transformation to points in the last frame to estimate camera motions, result is stable
    p1s_trans = cv.transform(np.array([p1s]), trans_m)[0] 
    #pure motion movement and noise
    delta = p2s - p1s_trans  
    #use mean value of magnitude to filter out large noise
    magnitudes = np.sqrt((delta ** 2).sum(-1))[..., np.newaxis] 
    mean = np.mean(magnitudes)
    mask = (magnitudes > mean*3)[:,0]
    mask2= (magnitudes <= 5)[:,0]
    mask = np.logical_or(mask,mask2)
    
    p1s = np.delete(p1s,mask,0)
    p2s = np.delete(p2s,mask,0)
    p1s_trans = np.delete(p1s_trans,mask,0)

    

    if down_sample and p1s.size>200: #take only 800 points
        n = int(p1s.size/200)
        p1s = p1s[0::n].copy()
        p2s = p2s[0::n].copy()
        p1s_trans = p1s_trans[0::n].copy()
    # if not moving only small amount of vectors will be extracted
    if p1s.size > 100:
        vecs =  (p2s - p1s_trans) 
        magnitudes = np.sqrt((vecs ** 2).sum(-1))[..., np.newaxis] 
        vecs*= np.sqrt(frame.shape[0]**2+frame.shape[1]**2)/magnitudes
        
        line1 = p2s + vecs
        line2 = p2s - vecs
        
        accumulator.fill(0)
        for i in range(len(p1s)):
            p1, p1_trans,p2 = p1s[i],p1s_trans[i],p2s[i]
            cv.circle(frame_trans_p2, p2.astype(int),2, (0,255,255) )
            # cv.arrowedLine(frame_trans_p2,p1.astype(int), p1_trans.astype(int), (0,255,0),1)
            cv.arrowedLine(frame_trans_p2, p1_trans.astype(int),p2.astype(int), (0,0,255),1)
            mask_accu.fill(0)
            p1, p2 = line1[i], line2[i]
            cv.line(mask_accu, p1.astype(int),p2.astype(int), (1),1)
            accumulator += mask_accu
        accumulator = cv.GaussianBlur(accumulator,(35,35),0) #most probablistic distribution
            
        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(accumulator, None)
        cv.circle(frame_trans_p2, maxLoc ,15, (0,0,255),-1 )

    last_frame = frame
    cv.imshow('frame', frame_trans_p2)
    cv.imshow('mask', accumulator)
    
    k = cv.waitKey(5)
    if k ==27:
        break
    

cap.release()
cv.destroyAllWindows()
