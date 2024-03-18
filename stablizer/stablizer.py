'''
Jiayou Qin 03/18/2024
SIFT based stablizer
a simple demo of stablization using sift
Used ransac to estimate affine transformation
'''


import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
video_path = "rotationTest.mp4" #change this file name as needed
cap = cv.VideoCapture(video_path)


visual_intensity = 1
visualize_points = False
visualize_cursor = True

for i in range(1): #skip frames, must be larger than 1
    success, frame = cap.read()
# frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
sift = cv.SIFT_create()
frame = cv.resize(frame,(800,600))
y,x = frame.shape[:2]
center = np.array([x/2,y/2], dtype=float)
cursor = center.copy()

last_frame = frame
kp, des = sift.detectAndCompute(frame,None)
cv.namedWindow('image',cv.WINDOW_NORMAL)
cv.namedWindow('trans',cv.WINDOW_NORMAL)
cv.imshow('image', frame)
cv.waitKey(0) #press key to begin


last_m = None
while True:
    for i in range(5):
        success, frame = cap.read()
    if not success:
        break
    angles = np.array([],dtype=float)
    delta = np.array([],dtype=float)
    p1s = np.array([],dtype=float)
    p2s = np.array([],dtype=float)
    visual_info = np.array([],dtype=float)
    frame = cv.resize(frame,(800,600))
#     frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    out_frame = frame.copy()
    if not success:
        break
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
            p1s = np.concatenate((p1s, p1))
            p2s = np.concatenate((p2s, p2))
            
    p1s = p1s.reshape(-1,2)
    p2s = p2s.reshape(-1,2)
    
    trans_m,_ = cv.estimateAffine2D(p1s,p2s)
    if last_m is not None:
        trans_m[:,:2] = np.matmul(trans_m[:,:2], last_m[:,:2])
        trans_m[:,2] += last_m[:,2]

    last_m = trans_m
    trans = cv.warpAffine(frame, trans_m, frame.shape[:2][::-1])
    cv.imshow('trans',trans)
    cv.imshow('image', out_frame)
    k = cv.waitKey(5)
    if k ==27:
        break
    
    last_frame = frame
    
#     break;
cap.release()
cv.destroyAllWindows()
