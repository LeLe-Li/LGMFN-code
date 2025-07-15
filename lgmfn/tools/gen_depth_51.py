import numpy as np
import cv2 #pip install opencv-python
import math
import os
from tqdm import tqdm


anchor = [3, 7, 11, 15, 19]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300
two_person_action = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]

def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def read_depth_xy(file, max_body=2, num_joint=25):  
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 2))   
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    if np.isnan(v['depthX']):
                        v['depthX'] = 0.0
                    if np.isnan(v['depthY']):
                        v['depthY'] = 0.0
                    data[m, n, j, :] = [v['depthX'], v['depthY']]
                else:
                    pass
    return data


def depth_box_vis_gray(ske_path, depth_path, gray_out_path, line, delta=48, limit=15):
    skeleton_ori_data = read_depth_xy(ske_path)    
    skeleton_data = skeleton_ori_data[:, :, anchor, :]
    num_people, num_frames, num_joints, num_dims = skeleton_data.shape
    frame_indices = np.linspace(0, num_frames-1, 5, dtype=int)
    
    frames = []
    for n_f in frame_indices:
        img = cv2.imread(depth_path + '/MDepth-{0:08d}.png'.format(n_f+1), -1)
        if np.sum(skeleton_data[1, :]) != 0 and (int(line[-3:]) not in two_person_action):
            person1height = np.max(skeleton_data[0, :, :, 1]) - np.min(skeleton_data[0, :, :, 1])
            person2height = np.max(skeleton_data[1, :, :, 1]) - np.min(skeleton_data[1, :, :, 1])
            if person1height > person2height:
                x_min = math.floor(np.min(skeleton_data[0, n_f, :, 0] - limit))
                x_max = math.floor(np.max(skeleton_data[0, n_f, :, 0] + limit))
                y_min = math.floor(np.min(skeleton_data[0, n_f, :, 1] - limit))
                y_max = math.floor(np.max(skeleton_data[0, n_f, :, 1] + limit)) 
            else:
                x_min = math.floor(np.min(skeleton_data[1, n_f, :, 0] - limit))
                x_max = math.floor(np.max(skeleton_data[1, n_f, :, 0] + limit))
                y_min = math.floor(np.min(skeleton_data[1, n_f, :, 1] - limit))
                y_max = math.floor(np.max(skeleton_data[1, n_f, :, 1] + limit))
            x_min = np.clip(x_min, 0, img.shape[1])
            x_max = np.clip(x_max, 0, img.shape[1])
            y_min = np.clip(y_min, 0, img.shape[0])
            y_max = np.clip(y_max, 0, img.shape[0])
            roi = img[y_min:y_max, x_min:x_max]
            roi_resized = cv2.resize(roi, (96, 480), interpolation=cv2.INTER_LINEAR)
            frames.append(roi_resized)
            
        else:                
            x_min = math.floor(np.min(skeleton_data[0, n_f, :, 0] - limit))
            x_max = math.floor(np.max(skeleton_data[0, n_f, :, 0] + limit))
            y_min = math.floor(np.min(skeleton_data[0, n_f, :, 1] - limit))
            y_max = math.floor(np.max(skeleton_data[0, n_f, :, 1] + limit))
            x_min = np.clip(x_min, 0, img.shape[1])
            x_max = np.clip(x_max, 0, img.shape[1])
            y_min = np.clip(y_min, 0, img.shape[0])
            y_max = np.clip(y_max, 0, img.shape[0])
            roi = img[y_min:y_max, x_min:x_max]
            roi_resized = cv2.resize(roi, (96, 480), interpolation=cv2.INTER_LINEAR)
            if np.sum(skeleton_data[1, :]) == 0:
                frames.append(roi_resized)               
            if np.sum(skeleton_data[1, :]) != 0:
                x_min2 = math.floor(np.min(skeleton_data[1, n_f, :, 0] - limit))
                x_max2 = math.floor(np.max(skeleton_data[1, n_f, :, 0] + limit))
                y_min2 = math.floor(np.min(skeleton_data[1, n_f, :, 1] - limit))
                y_max2 = math.floor(np.max(skeleton_data[1, n_f, :, 1] + limit))
                x_min2 = np.clip(x_min2, 0, img.shape[1])
                x_max2 = np.clip(x_max2, 0, img.shape[1])
                y_min2 = np.clip(y_min2, 0, img.shape[0])
                y_max2 = np.clip(y_max2, 0, img.shape[0])
                roi2 = img[y_min2:y_max2, x_min2:x_max2]
                roi2_resized = cv2.resize(roi2, (48, 480), interpolation=cv2.INTER_LINEAR)            
                img_stitched = np.hstack((roi_resized, roi2_resized))
                frames.append(img_stitched)
               
        combined_image = np.concatenate(frames, axis=1)
        combined_image_resized = cv2.resize(combined_image, (480, 480), interpolation=cv2.INTER_LINEAR)
        combined_image_resized = combined_image_resized.astype(np.uint8)

        np.save(os.path.join(gray_out_path, f'{line.strip()}.npy'), combined_image_resized)


if __name__ == '__main__':
    with open('/home/user/ntu60.txt', 'r') as file:
        lines = file.readlines()
        for line in tqdm(lines, desc="Processing lines", unit="line"):
            ske_path = f'/home/user/nturgb+d_skeletons/{line.strip()}.skeleton'
            dep_path = f'/home/user/nturgb+d_depth_masked/{line.strip()}'
            gray_out_path = f'/home/user/depth_51_npygen/'
            depth_box_vis_gray(ske_path, dep_path, gray_out_path, line.strip())

