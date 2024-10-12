# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 23:25:25 2024

@author: MaxGr
"""


import os
import time
import numpy as np
from collections import deque
from datetime import datetime
import cv2
from helper_functions import *


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"
current_datetime = datetime.now()
date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


print(cv2.__version__)
cv2_cuda_enable = cv2.cuda.getCudaEnabledDeviceCount()
print(cv2_cuda_enable) # Check if CUDA is available in OpenCV

input_folder = 'dataset'
# file_list = os.listdir('./annotation/')
file_list = os.listdir(input_folder)

supported_formats = ['mp4','mov']
for file_name in file_list:
    if file_name.split('.')[-1].lower() not in supported_formats:
        continue
    print(f'reading file: {file_name}')
    
    input_file = os.path.join(input_folder, file_name)
    cap = cv2.VideoCapture(input_file)
    
    gifs = []
    
    image_size = 512
    skip_frames = 4
    start_frame = 1
    end_frame = 10000
    output = False
    output_size = (1024,512)
    frame_list = deque()
    frame_time = []
    attention_list = deque()
    motion_center_list = deque()
    
    # Create a VideoWriter for the output video
    if output:
        os.makedirs('output', exist_ok=True)
        output_video_filename = f"./output/{file}_{date_time_string}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        output_video = cv2.VideoWriter(output_video_filename, fourcc, 15, output_size)
        
    
    # Capture video
    for i in range(start_frame): #skip frames, must be larger than 1
        ret, frame1 = cap.read()
    img_height, img_width = frame1.shape[:2]
    
    # Convert the first frame to grayscale
    frame1 = img_resize(frame1, image_size)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # sift = cv2.SIFT_create()
    # kp2, des2 = sift.detectAndCompute(gray1,None)
    frame_list.append(frame1)
    
    if cv2_cuda_enable:
        gpu_previous = cv2.cuda_GpuMat()
        gpu_current= cv2.cuda_GpuMat()
        gpu_flow = cv2.cuda_FarnebackOpticalFlow.create()
        optical_flow = cv2.cuda_GpuMat()
        
    mask_accu = np.zeros((image_size,image_size), dtype=float)
    accumulator = np.zeros((image_size,image_size), dtype=float)
    motion_heatmap = np.zeros_like(frame1)
    
    annotation = []
    
    frame_id = 0
    # Read frames from the video
    while cap.isOpened():
        # print(frame_id)
        frame_id += 1
    
        for i in range(skip_frames): 
            frame_id += 1
            ret, frame = cap.read()
            failed = False
            if not ret:
                failed = True
                continue
        if failed:
            break
    
        if frame_id > end_frame: break
    
        
        frame2 = img_resize(frame, image_size)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        motion_heatmap.fill(0)
    
        if cv2_cuda_enable:
            GPU_time = time.time()
            # Create cuda matrix
            gpu_previous.upload(gray1)
            gpu_current.upload(gray2)
            
            # Calculate Optical Flow on GPU
            flow_time = time.time()
            img = gpu_flow.calc(gpu_previous, gpu_current, optical_flow, None)
            flow_time = time.time() - flow_time
            flow = img.download()
            GPU_time = time.time() - GPU_time
        else:
            flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        thresh_low  = np.mean(magnitude)*1
        thresh_high = np.mean(magnitude)*50
    
        u = flow[..., 0]  # Horizontal component of flow
        v = flow[..., 1]  # Vertical component of flow
        h, w = u.shape
        x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
        

        grid = np.moveaxis(np.mgrid[:image_size,:image_size], 0, -1)
        grid = np.flip(grid, axis=2)
        grid_prev = grid-flow
        magnitudes = np.sqrt((flow ** 2).sum(-1))[..., np.newaxis]
        mask = (magnitudes < np.mean(magnitudes)*0.5)[:,:,0] #note that some patch will not have optical flow like the sky
        
        # SVD_time = time.time()
        kp_time = time.time()
        
        k1s = grid_prev.reshape(1,-1,2)[0][::100,:]#.shape
        k2s = grid.reshape(1,-1,2)[0][::100]
        
        kp_time = time.time() - kp_time

        
        match_time = time.time()
        
        mat, [U, S, Vt] = SVD_keypoint(np.array(k1s), np.array(k2s))
        
        match_time = time.time() - match_time
        mat = mat[:2]
        # mat = transform_matrix_reverse[:2]
        stabilized_gray1 = cv2.warpAffine(gray1, mat, (image_size, image_size))
        stabled = cv2.addWeighted(gray2, 0.5, stabilized_gray1, 0.5, gamma=0)
        stabled_RGB = cv2.cvtColor(stabled, cv2.COLOR_GRAY2RGB)
        if cv2_cuda_enable:
            # Calculate Motion Flow on GPU
            gpu_previous.upload(stabilized_gray1)
            gpu_current.upload(gray2)
            img = gpu_flow.calc(gpu_previous, gpu_current, optical_flow, None)
            motion_flow = img.download()
        else:
            motion_flow = cv2.calcOpticalFlowFarneback(stabilized_gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        
        noise = cv2.transform(grid_prev.reshape(1,-1,2), mat) #camera movement
        noise = noise.reshape(image_size,image_size,2) - grid_prev
        epsilon, _ = cv2.cartToPolar(noise[..., 0], noise[..., 1])

        noise[mask,:] = np.array([0,0])

        motion, _ = cv2.cartToPolar(motion_flow[..., 0], motion_flow[..., 1])

        # plt.imshow(motion)
        step = 6
        p1s = np.array(k1s[::step]).reshape(-1, 2)
        p2s = np.array(k2s[::step]).reshape(-1, 2)
        # Draw motion vector
        p2s_trans = cv2.transform(np.array([p2s]), cv2.invertAffineTransform(mat))[0] 

        p1s_draw = np.array(p1s).reshape(-1, 2).astype(int)
        p2s_draw = np.array(p2s_trans).reshape(-1, 2).astype(int)
            
        for i in range(len(p1s_draw)):
            p1 = p1s_draw[i]
            p2 = p2s_draw[i]
            cv2.arrowedLine(motion_heatmap, p1, p2, (0, 255, 0), 1)
    
        # if p1s.size > 50:
        # Calculate vectors, magnitudes, and normalize vectors
        vecs = p1s - p2s_trans
        
        magnitudes = np.sqrt((vecs ** 2).sum(axis=-1))[..., np.newaxis]
        vecs*= np.sqrt(image_size**2 + image_size**2) / magnitudes
        mean_magnitude = np.mean(magnitudes)
        mask = (magnitudes < mean_magnitude*1)[:,0]
    
        # p2s = np.array(p1s)
        p1s = p1s[~mask] 
        p2s = p2s[~mask] 
        vecs = vecs[~mask] 
    
        # Calculate line endpoints
        line1 = (p1s + vecs).astype(int)
        line2 = (p1s - vecs).astype(int)
    
        # Draw lines and update accumulator
        accumulator.fill(0)
        h,w = accumulator.shape[:2]
        for p1, p2 in zip(line1, line2):
            mask = get_line_pixels(w, h, p1[0], p1[1], p2[0], p2[1])
            accumulator[mask[:, 1], mask[:, 0]] += 1
        # use gaussian since the vector is a probability distribution
        accumulator = cv2.GaussianBlur(accumulator, (51, 51), 0)
        # Find maxLoc
        _, _, _, maxLoc = cv2.minMaxLoc(accumulator)
    
        acceration = 1/(int(mean_magnitude)+1) * 50 * skip_frames
        attention_map = generate_gaussian_image(image_size, image_size, maxLoc, acceration)
        attention_list.append(attention_map)
        motion_center_list.append([maxLoc, acceration])

        # note that attention map is smoothened on time domain(accumulation of frames)
        if len(attention_list) > 10:
            # print(len(attention_list))
            attention_map = np.sum(attention_list, 0)#.astype(np.uint8)
            attention_list.popleft()
            for center_i, size in motion_center_list:
                cv2.circle(stabled_RGB, center_i, int(size), (0, 0, 255), 5)
            motion_center_list.popleft()
        
        annotation.append([frame_id, maxLoc, kp_time, match_time])
        
        attention_map = cv2.applyColorMap(img_uint8(attention_map), cv2.COLORMAP_JET)
        attention_frame = cv2.addWeighted(frame2, 0.2, attention_map, 0.8, gamma=0)
        accumulator_map = cv2.cvtColor(img_uint8(accumulator), cv2.COLOR_GRAY2RGB)
        epsilon_map = cv2.applyColorMap(img_uint8(epsilon), cv2.COLORMAP_PLASMA)
        op_flow = cv2.applyColorMap(img_uint8(magnitude), cv2.COLORMAP_PLASMA)
        motion_flow = cv2.applyColorMap(img_uint8(motion), cv2.COLORMAP_PLASMA)
        motion_flow_vector = cv2.applyColorMap(motion_heatmap, cv2.COLORMAP_HOT)
    
        motor_attention = np.hstack((stabled_RGB, attention_frame))
        flow_map = np.hstack((op_flow, epsilon_map, motion_flow))
        vector_map = np.hstack((motion_heatmap, accumulator_map))

        h_cat1 = np.hstack((frame2, epsilon_map, stabled_RGB, attention_frame))
        h_cat2 = np.hstack((op_flow, motion_flow, accumulator_map, motion_heatmap))
        v_cat = np.vstack((h_cat1, h_cat2))
        
        cv2.putText(v_cat, f'{frame_id}',(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f'{input_file}', v_cat)
        
        v_cat = cv2.resize(v_cat, output_size)

        if output:
            output_video.write(v_cat)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print('Frame saved...')
            cv2.imwrite(f'./plot/grid_{frame_id}.png', v_cat)

        if key == ord('q'):
            break
    
        # Update previous frame and its grayscale
        gray1 = gray2
    
    # Release video capture and writer
    cap.release()
    cv2.destroyAllWindows()
    if output: 
        output_video.release()








