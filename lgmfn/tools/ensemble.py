# -*- coding: utf-8 -*-


import pickle
import numpy as np
from tqdm import tqdm

label_data = open('pgmfn//data//ntu120//xset//val_label.pkl', 'rb')
label = np.array(pickle.load(label_data))

# msg3d_score_pkl
r1 = open('final-resnet//120//xset//joint-posec3d.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('final-resnet//120//xset//bone-posec3d.pkl', 'rb')
r2 = list(pickle.load(r2).items())

# hci_score_pkl
# r3 = open('ntu60/xsub/rgb/res18-778.pkl', 'rb')
# r3 = list(pickle.load(r3).items())
# r4 = open('ntu60/xsub/rgb-mot/res18-698.pkl', 'rb')
# r4 = list(pickle.load(r4).items())

r3 = open('pgmfn-results-effi//rgb51-xset-effi.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('pgmfn-results-effi//rgb55-xset-effi.pkl', 'rb')
r4 = list(pickle.load(r4).items())

r5 = open('pgmfn-results-effi//depth51-xset-effi.pkl', 'rb')
r5 = list(pickle.load(r5).items())
r6 = open('pgmfn-results-effi//depth55-xset-effi.pkl', 'rb')
r6 = list(pickle.load(r6).items())

right_num = total_num = 0
for i in tqdm(range(len(label[0]))):
    _, l = label[:, i]
    _, r11 = r1[i]
    _, r22 = r2[i]
    _, r33 = r3[i]
    _, r44 = r4[i]
    _, r55 = r5[i]
    _, r66 = r6[i]
     
    exp_3 = np.exp(r33)
    softmax_3 = exp_3 / np.sum(exp_3)

    exp_4 = np.exp(r44)
    softmax_4 = exp_4 / np.sum(exp_4)
    
    exp_5 = np.exp(r55)
    softmax_5 = exp_5 / np.sum(exp_5)

    exp_6 = np.exp(r66)
    softmax_6 = exp_6 / np.sum(exp_6)

    
    #alpha = [0, 0, 0.75, 1, 0.75, 0.3] #RL+RG+DL+DG four streams csub efficientnet
    #alpha = [0, 0, 1, 1, 0.8, 0.3] #four streams cview
    #alpha = [0, 0, 0.9, 1, 1.1, 0.8] #four streams xsub
    #alpha = [0, 0, 1, 1, 1, 0.6] #four streams xset
    
    
    #r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3] + r55 * alpha[4] + r66 * alpha[5] #MS-G3D choose this
    r = r11 * alpha[0] + r22 * alpha[1] + softmax_3 * alpha[2] + softmax_4 * alpha[3] + softmax_5 * alpha[4] + softmax_6 * alpha[5] #to use PoseC3D one should go through a softmax

    r = np.argmax(r)
    right_num += int(r == int(l))
    total_num += 1
acc = right_num / total_num

print('\n', acc*100)