# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 23:25:25 2024

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import time
import math
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from datetime import datetime

current_datetime = datetime.now()
date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")


import cv2
print(cv2.__version__)
cv2_cuda_enable = cv2.cuda.getCudaEnabledDeviceCount()
print(cv2_cuda_enable) # Check if CUDA is available in OpenCV


def img_uint8(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image

def img_resize(image, size):
    height, width = image.shape[:2]
    crop_size = min(height, width)
    start_y = (height - crop_size) // 2
    start_x = (width - crop_size) // 2
    center_cropped_image = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    center_cropped_image = cv2.resize(center_cropped_image,(size, size))
    return center_cropped_image


def generate_gaussian_image(height, width, center_point, sigma):
    # Create a grid of coordinates
    (center_x, center_y) = center_point
    x = np.arange(width) - center_x
    y = np.arange(height) - center_y
    x, y = np.meshgrid(x, y)

    # Calculate the squared distances from the center
    distances_squared = x ** 2 + y ** 2

    # Calculate the Gaussian kernel
    kernel = np.exp(-distances_squared / (2 * sigma ** 2))

    # Normalize the kernel to the range [0, 1]
    kernel_normalized = (kernel - np.min(kernel)) / (np.max(kernel) - np.min(kernel))

    return kernel_normalized


def SVD_keypoint(keypoint_list_1, keypoint_list_2):
    center_1 = np.mean(keypoint_list_1, axis=0)
    center_2 = np.mean(keypoint_list_2, axis=0)

    new_keypoint_list_1 = keypoint_list_1 - center_1
    new_keypoint_list_2 = keypoint_list_2 - center_2

    M = new_keypoint_list_2.T @ new_keypoint_list_1
    u, s, vt = np.linalg.svd(M)

    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1  # Adjust the last column of u
        R = u @ vt

    T = center_2 - R @ center_1

    # Create a homogeneous transformation matrix
    transform_matrix = np.eye(3)  # Start with an identity matrix
    transform_matrix[0:2, 0:2] = R  # Insert R into the top-left
    transform_matrix[0:2, 2] = T   # Insert T into the top-right

    return transform_matrix, [u, s, vt]





path = './Videos/'
# file = 'NJ_1_1.mp4'
# file = 'NJ_1_2.mp4'
# file = 'NJ_1_3.mp4'
# file = 'NJ_1_5.mp4'
# file = 'NJ_2_1.mp4'
# file = 'NJ_3_1.mp4'
# file = 'NJ_4_1.mp4'
# file = 'NJ_4_2.mp4'
# file = 'NJ_5_1.mp4'
# file = 'NJ_5_2.mp4'
# file = 'NJ_6_2.mp4'
# file = 'NJ_6_3.mp4'
# file = 'NJ_7_2.mp4'

# file = 'JP_1.MOV'
# file = 'JP_4.MOV'
# file = 'JP_5.MOV'
# file = 'JP_12.MOV'


# file = 'SC_2.MOV'
# file = 'SC_5.MOV'
# file = 'SC_6.MOV'
# file = 'SC_7.MOV'
# file = 'SC_10.MOV' 
# file = 'SC_11.MOV' 

# file = 'rotationTest.mp4'



# path = './Dataset/'
# path = 'D:/Data/H_detection/Dataset/'

# file = 'Chongqing Drive.mp4'
# file = 'Central London Sunset Walk  Relaxing Evening Walk through West End [4K HDR].mp4'
# file = '[4K]China walk tour  Thursday Walk in ChunXi Road Chengdu  Real china city.mp4'
# file = 'Driving in Switzerland 6 From Grindelwald to Lauterbrunnen  4K 60fps.mp4'
# file = 'Seoul KOREA - Myeongdong Shopping Street [Travel Vlog].mp4'
# file = 'City Walks - Venice Italy Walking Tour and Virtual Treadmill Walk.mp4'
# file = '4k hdr japan travel  Walk in Kamakura Kanagawa Japan   Relaxing Natural City ambience.mp4'


import imageio

# file_list = os.listdir('./annotation/')
file_list = os.listdir(path)
file_list.remove('desktop.ini')

for name in file_list:
    file = name#[:-4]
    input_file = path + file
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
        # if frame_id == 200: break
        
    
        # ret, frame = cap.read()
        # frame_list.append(frame)
        # if not ret:
        #     break
        
        # if frame_id <= skip_frames:
        #     continue
        
        # if frame_id > skip_frames:
        #     # print(frame_id, len(frame_list))
            
        #     frame1 = frame_list[0]
        #     gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        #     gray1 = img_resize(gray1, image_size)
        #     # sift = cv2.SIFT_create()
        #     # kp1, des1 = sift.detectAndCompute(gray1,None)
        #     frame_list.popleft()
        #     frame = frame_list[-1]
    
        
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
        # magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # # magnitude = cv2.GaussianBlur(magnitude, (201, 201), 0) 
        # # magnitude = cv2.blur(magnitude, (101, 101)) 
        # cv2.imshow('magnitude', magnitude)
    
        # magnitudes = np.sqrt((flow ** 2).sum(-1))[..., np.newaxis]
        thresh_low  = np.mean(magnitude)*1
        thresh_high = np.mean(magnitude)*50
        # mask = (magnitude < threshVal) #note that some patch will not have optical flow like the sky
    
    
        # if frame_id % 5 == 0:
        #     total_time = np.array([flow_time, GPU_time])*1000
        #     frame_time.append(total_time)
        #     print(total_time)
    
    
        u = flow[..., 0]  # Horizontal component of flow
        v = flow[..., 1]  # Vertical component of flow
        h, w = u.shape
        x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
        
        
        p1s = []
        p2s = []
    
        step = 20
        for y in range(0, image_size, step):
            for x in range(0, image_size, step):
                dx = u[y, x]
                dy = v[y, x]
                pt1 = (x, y)
                pt2 = (x + dx), (y + dy)
    
                # if thresh_low < (dx**2 + dy**2) < thresh_high:
                p1s.append(pt1)
                p2s.append(pt2)
            
                # pt2_draw = (pt2[0].astype(int),pt2[1].astype(int))
                # cv2.arrowedLine(motion_heatmap, pt1, pt2_draw, (0, 255, 0), 1)
            
    
            
        # mask = (magnitude < threshVal) #note that some patch will not have optical flow like the sky
        # mat = cv2.estimateAffine2D(grid_prev.reshape(1,-1,2)[::50], grid.reshape(1,-1,2)[::50])[0]
    
        transform_matrix,_ = cv2.estimateAffine2D(np.array(p2s), np.array(p1s))
        stabilized_gray2 = cv2.warpAffine(gray2, transform_matrix, (image_size, image_size))
    
        transform_matrix_reverse,_ = cv2.estimateAffine2D(np.array(p1s), np.array(p2s))
        stabilized_gray1 = cv2.warpAffine(gray1, transform_matrix_reverse, (image_size, image_size))
    
        # cv2.imshow('motion_heatmap', motion_heatmap)
    
    
    
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

        
        
        # u = motion_flow[..., 0]  # Horizontal component of flow
        # v = motion_flow[..., 1]  # Vertical component of flow
        # h, w = u.shape
        # x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))
        
        
        # p1s = []
        # p2s = []
    
        # step = 20
        # for y in range(0, image_size, step):
        #     for x in range(0, image_size, step):
        #         dx = u[y, x]
        #         dy = v[y, x]
        #         pt1 = (x, y)
        #         pt2 = (x + dx), (y + dy)
    
        #         # if thresh_low < (dx**2 + dy**2) < thresh_high:
        #         p1s.append(pt1)
        #         p2s.append(pt2)
            
        #         pt2_draw = (pt2[0].astype(int),pt2[1].astype(int))
        #         cv2.arrowedLine(motion_heatmap, pt1, pt2_draw, (0, 255, 0), 1)
            
    
        
    
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


        mat = transform_matrix_reverse[:2]
        
        
        
        # mat = cv2.estimateAffine2D(grid_prev.reshape(1,-1,2)[::100], grid.reshape(1,-1,2)[::100])[0]
        noise = cv2.transform(grid_prev.reshape(1,-1,2), mat) #camera movement
        noise = noise.reshape(image_size,image_size,2)-grid_prev
        epsilon, _ = cv2.cartToPolar(noise[..., 0], noise[..., 1])

        # SVD_time = time.time() - SVD_time
        # print(SVD_time*1000)
        noise[mask,:] = np.array([0,0])
        # flow[mask,:] =  np.array([0,0])
        # motion_flow[mask,:] =  np.array([0,0])
        # motion_flow = flow-noise
        
        # motion_center = -np.mean(motion_flow.reshape(1,-1,2)[0][::100], 0)*20 + [image_size/2, image_size/2]
        # print(motion_center)


        motion, _ = cv2.cartToPolar(motion_flow[..., 0], motion_flow[..., 1])
        # motion, _ = cv2.cartToPolar(motion_flow[..., 0], motion_flow[..., 1])
        # motion = cv2.GaussianBlur(motion, (301, 301), 0)
        # motion = cv2.normalize(motion, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # cv2.imshow('motion', motion)
        # mask = (motion < np.mean(motion)*0.25) #note that some patch will not have optical flow like the sky
        # motion[mask] = 0
        
        # plt.imshow(motion)
        p1s = np.array(p1s).reshape(-1, 2)
        p2s = np.array(p2s).reshape(-1, 2)
            
        # Draw motion vector
        p2s_trans = cv2.transform(np.array([p2s]), transform_matrix)[0] 
        
        
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
        for p1, p2 in zip(line1, line2):
            mask_accu.fill(0)
            cv2.line(mask_accu, tuple(p1), tuple(p2), (1), 1)
            # cv2.line(stabled_RGB, tuple(p1), tuple(p2), (1), 1)
            accumulator += mask_accu
    
        # Blur accumulator
        accumulator = cv2.GaussianBlur(accumulator, (51, 51), 0)
        

        
    
    
        # Find maxLoc
        _, _, _, maxLoc = cv2.minMaxLoc(accumulator)
        # maxLoc = motion_center.astype(int)
        # cv2.imshow('frame2', frame2)
    
    
    
        acceration = 1/(int(mean_magnitude)+1) * 50 * skip_frames
        attention_map = generate_gaussian_image(image_size, image_size, maxLoc, acceration)
        # attention_map = generate_gaussian_image(image_size, image_size, maxLoc, 100)
        
        attention_list.append(attention_map)
        motion_center_list.append([maxLoc, acceration])

        if len(attention_list) > 10:
            # print(len(attention_list))
            attention_map = np.sum(attention_list, 0)#.astype(np.uint8)
            # attention_map = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            attention_list.popleft()
            
            for center_i, size in motion_center_list:
                cv2.circle(stabled_RGB, center_i, int(size), (0, 0, 255), 5)
            
            motion_center_list.popleft()
            
        # _, _, _, maxLoc = cv2.minMaxLoc(attention_map)
    
    
        # centroid_y, centroid_x = np.mean(np.where(attention_map==np.max(attention_map)),1).astype(int)
        # attention_point = (centroid_x, centroid_y)
        annotation.append([frame_id, maxLoc, kp_time, match_time])
        
        # cv2.circle(stabled_RGB, attention_point, 50, (0, 0, 255), 10)
    
    
    
        # attention_map = cv2.cvtColor(img_uint8(attention_map), cv2.COLOR_GRAY2RGB)
        attention_map = cv2.applyColorMap(img_uint8(attention_map), cv2.COLORMAP_JET)
        attention_frame = cv2.addWeighted(frame2, 0.2, attention_map, 0.8, gamma=0)
        
        # # Draw triangles
        # w1 = 256
        # w2 = 128
        # p1 = (256 - w1, image_size)
        # p2 = (256 + w1, image_size)
        # p3 = (256 - w2, image_size)
        # p4 = (256 + w2, image_size)
        # triangle_cnt = np.array([[p1, p2, maxLoc]])
        # triangle_cnt2 = np.array([[p3, p4, maxLoc]])
        # # cv2.drawContours(frame2, triangle_cnt, 0, (0, 255, 0), 3)
        # # cv2.drawContours(frame2, triangle_cnt2, 0, (0, 255, 255), 3)
        
        # H_1 = (maxLoc[0] - 50, maxLoc[1]+100)
        # H_2 = (maxLoc[0] + 50, maxLoc[1]+100)
    
        # cv2.line(frame2, H_1, H_2, (0, 255, 0), 2)
        # cv2.line(frame2, H_1, p1, (0, 255, 0), 2)
        # cv2.line(frame2, H_2, p2, (0, 255, 0), 2)
        
        # # cv2.drawContours(frame2, np.array([[H_1, p1, p2]]), 0, (0, 255, 255), 3)
    
        # midpoint_x = (maxLoc[0] + image_size//2) // 2
        # midpoint_y = (maxLoc[1] + image_size) // 2
        
        # M_1 = (midpoint_x - 150, midpoint_y+50)
        # M_2 = (midpoint_x + 150, midpoint_y+50)
      
        # cv2.line(frame2, M_1, M_2, (0, 255, 255), 3)
        # cv2.line(frame2, M_1, p1, (0, 255, 255), 3)
        # cv2.line(frame2, M_2, p2, (0, 255, 255), 3)
    
    
    
    
        accumulator_map = cv2.cvtColor(img_uint8(accumulator), cv2.COLOR_GRAY2RGB)
    
        epsilon_map = cv2.applyColorMap(img_uint8(epsilon), cv2.COLORMAP_PLASMA)
    
        op_flow = cv2.applyColorMap(img_uint8(magnitude), cv2.COLORMAP_PLASMA)
    
        motion_flow = cv2.applyColorMap(img_uint8(motion), cv2.COLORMAP_PLASMA)
        motion_flow_vector = cv2.applyColorMap(motion_heatmap, cv2.COLORMAP_HOT)
        # blended = cv2.addWeighted(frame2, 0.2, motion_heatmap, 0.8, gamma=0)
        
        # stabled_RGB[:,:,1:].fill(0)
        # motion_stabled = cv2.add(stabled_RGB, motion_SIFT)
    
        motor_attention = np.hstack((stabled_RGB, attention_frame))
        flow_map = np.hstack((op_flow, epsilon_map, motion_flow))
        vector_map = np.hstack((motion_heatmap, accumulator_map))

        h_cat1 = np.hstack((frame2, epsilon_map, stabled_RGB, attention_frame))
        h_cat2 = np.hstack((op_flow, motion_flow, accumulator_map, motion_heatmap))
        v_cat = np.vstack((h_cat1, h_cat2))
        
        
    
        cv2.putText(v_cat, f'{frame_id}',(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f'{file}', v_cat)
    
        # total_time = np.array([SIFT_time, match_time])*1000
        # print(total_time)
        
        v_cat = cv2.resize(v_cat, output_size)
        # gifs.append(v_cat)

        if output:
            output_video.write(v_cat)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            print('Frame saved...')
            cv2.imwrite(f'./plot/grid_{frame_id}.png', v_cat)
            # cv2.imwrite(f'./plot/gray1_{frame_id}.png', gray1)
            # cv2.imwrite(f'./plot/gray2_{frame_id}.png', gray2)
            # cv2.imwrite(f'./plot/motor_attention_{frame_id}.png', motor_attention)
            # cv2.imwrite(f'./plot/flow_map_{frame_id}.png', flow_map)
            # cv2.imwrite(f'./plot/vector_map_{frame_id}.png', vector_map)


    
        if key == ord('q'):
            break
    
        # Update previous frame and its grayscale
        gray1 = gray2
    
    # Release video capture and writer
    cap.release()
    cv2.destroyAllWindows()
    if output: output_video.release()

    # imageio.mimsave(f'./output/{file}_{date_time_string}.gif', gifs)

    # save_file = np.array([file, 
    #                       frame_id, 
    #                       (image_size, image_size), 
    #                       annotation], dtype=object)
    
    
    # np.save(f'./test_data/our/{file}.npy', save_file)
    
    






        # cv2.imwrite(f'./plot/blended_{frame_id}.png', blended)
        # cv2.imwrite(f'./plot/frame_{frame_id}.png', frame2)
        # cv2.imwrite(f'./plot/heatmap_{frame_id}.png', heatmap)


    # cv2.imshow('gray', gray1)
    # cv2.imshow('motion_SIFT', motion_SIFT)
    # cv2.imshow('frame2', frame2)
    # cv2.imshow('motion flow', motion_flow)
    # cv2.imshow('motion flow vector', blended)
    # cv2.imshow('motion stabled', stabled_RGB)
    # cv2.imshow('blended', blended)
    
    

    # # Use mean value of magnitude to filter out large noise
    # magnitudes = np.sqrt((delta ** 2).sum(-1))[..., np.newaxis] 
    # mean = np.mean(magnitudes)
    # mask = (magnitudes > mean*2)[:,0]
    # mask2 = (magnitudes < mean*0.5)[:,0]
    # mask = np.logical_or(mask,mask2)
    # p1s = np.delete(p1s,mask,0)
    # p2s = np.delete(p2s,mask,0)
    # p1s_trans = np.delete(p1s_trans,mask,0)


# flow = cv2.cuda_GpuMat()
# flow.calc(gpu_previous, gpu_current, flow, None)

# gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc( gpu_previous, gpu_current, gpu_flow, None )
# flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# pyramid_depth, 
# pyr_scale=0.5
# False,
# window_size, 
# iterations, 
# poly_n, 
# poly_sigma, 0
# gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False,15, 10, 5, 1.1, 0)
# gpu_flow = cv2.cuda_FarnebackOpticalFlow.calc( gpu_previous, gpu_current, flow, None )






        
    # grid = np.moveaxis(np.mgrid[:flow.shape[0],:flow.shape[1]], 0, -1)
    # grid_prev = grid-flow
    # # mat = cv2.estimateAffine2D(grid_prev.reshape(1,-1,2), grid.reshape(1,-1,2))[0]
    # # print(mat)
    
    
    # p1s = grid_prev.reshape(1,-1,2)[0]
    # p2s = grid.reshape(1,-1,2)[0]
       
    
    # # transform_matrix_reverse, [U, S, Vt] = SVD_keypoint(np.array(p1s), np.array(p2s))
    # # transform_matrix_reverse = transform_matrix_reverse[:2]
    
    # # SVD_keypoint(np.array(p2s), np.array(p1s))
    # # A = []
    # # B = []
    # # for (x, y), (xp, yp) in zip(p1s, p2s):
    # #     A.append([x, y, 1, 0, 0, 0])
    # #     A.append([0, 0, 0, x, y, 1])
    # #     B.append(xp)
    # #     B.append(yp)
    
    # # A = np.array(A)
    # # B = np.array(B)
    
    # # Apply SVD
    # # U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # # x = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ B
    
    # # Reshape x into 2x3 matrix
    # # affine_transform = x.reshape(2, 3)
    # # stabilized_gray2 = cv2.warpAffine(gray2, affine_transform, (image_size, image_size))

        
    # # y = 512
    # # x = 512
    # # grid = np.moveaxis(np.mgrid[:y,:x], 0, -1)
    # # grid.shape
    
    # # np.matmul(grid, transform_matrix[:,:2])


    # grid = np.moveaxis(np.mgrid[:image_size,:image_size], 0, -1)
    # grid_prev = grid-flow
    
    
    # noise = cv2.transform(grid_prev.reshape(1,-1,2), transform_matrix) #camera movement
    # noise = noise.reshape(image_size, image_size, 2)
    # smoothened = grid-noise















