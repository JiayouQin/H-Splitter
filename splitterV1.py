import torch
import torchvision.io
import torch.nn.functional as F

import cv2 as cv
import numpy as np
# from torchvision.utils import flow_to_image
import torchvision.transforms as T
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


transform = T.Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])
    
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = 'vits' # can also be 'vitb' or 'vitl'
model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).to(device)


# Open the video file
video_path = "../gpt/test1.mp4"
cap = cv.VideoCapture(video_path)

# Store the track history
cv.namedWindow("image", cv.WINDOW_NORMAL)
cv.namedWindow("depth", cv.WINDOW_NORMAL)
failed = False

sift = cv.SIFT_create()

down_sample = True

for i in range(60): #skip frames, must be larger than 1
    success, frame = cap.read()
last_frame = cv.resize(frame,(800,600))
kp, des = sift.detectAndCompute(frame,None)

mask_accu = np.zeros(last_frame.shape[:2],dtype=float)
accumulator = np.zeros(last_frame.shape[:2],dtype=float) #due to voting scheme this data type should be large
depthMask = np.zeros(last_frame.shape[:2],dtype=np.uint8)
while True:
    skip_frames = 5  #skip frames
    for i in range(skip_frames): 
        success, frame = cap.read()
        failed = False
        if not success:
            print('failed')
            failed = True
            continue
        frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    if failed:
        break
        #processing starts here
    frame = cv.resize(frame,(800,600))
    frameSIFT = frame.copy()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) / 255.0
    image = transform({'image': frame})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.no_grad():
        depth = model(image)
    h, w = frame.shape[:2]
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) #normalization, we disable this for accuracy
    depth = depth.cpu().numpy()


    angles = np.array([],dtype=float)
    delta = np.array([],dtype=float)
    p1s = np.array([],dtype=float)
    p2s = np.array([],dtype=float)
    kp_,des_ = kp,des
    kp, des = sift.detectAndCompute(frameSIFT,None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des,des_,k=2)
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
            mask_accu.fill(0)
            p1, p2 = line1[i], line2[i]
            cv.line(mask_accu, p1.astype(int),p2.astype(int), (1), 1)
            accumulator += mask_accu
        accumulator = cv.GaussianBlur(accumulator,(35,35),0) #most probablistic distribution

            
        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(accumulator, None)
        cv.circle(frameSIFT, maxLoc ,15 , (0,0,255),-1 )
        
        w1 = 300
        w2 = 150
        p1 = (400-w1,600)
        p2 = (400+w1,600)
        p3 = (400-w2,600)
        p4 = (400+w2,600)
        triangle_cnt = np.array( [[p1,p2,maxLoc]] )
        triangle_cnt2 = np.array( [[p3,p4,maxLoc]] )
        cv.drawContours(frameSIFT, triangle_cnt, 0, (0,255,0),3)
        cv.drawContours(frameSIFT, triangle_cnt2, 0, (0,255,255),3)
        # cv.line(frameSIFT, maxLoc, , (0,255,0),3)
        # cv.line(frameSIFT, maxLoc, (400+w,600), (0,255,0),3)
        
        #depth editing
        # depthMask.fill(1)
        # cv.drawContours(depthMask, triangle_cnt, 0, (0),-1)
        # depth[depthMask==1] = 0
        # depth[depth>5] = 0
        
    
    last_frame = frameSIFT
    
    
    cv.imshow('image', frameSIFT)
    cv.imshow('depth', depth )

    k = cv.waitKey(5)
    if k & 0xFF == 27:
        break

# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()
