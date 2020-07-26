import os
import numpy as np
from sklearn import preprocessing

# Get N data (-1 for all) 
def get_N_data(N, data_list):
    if N == -1:
        return data_list
    assert type(N) == int and N > 0
    assert len(data_list) > 0
    data_list.sort()
    if N <= len(data_list):
        data_list_N = data_list[::int(len(data_list)/N)][:N]
    else:
        data_list_N = []
        for data in data_list:
            for i in range(int(N/len(data_list))):
                data_list_N.append(data)
            if data_list.index(data) < N-int(N/len(data_list))*len(data_list):
                data_list_N.append(data)
    return data_list_N

# Get training and verification features
def get_data(feat_path, gt_path, feat, feat_len_keep):
    with open(gt_path) as fin:
        gt_dict = {}
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            gt_dict[line.split(' ')[0]] = line.split(' ')[1]
    
    x = []
    y = []
    for seg in gt_dict:
        seg_frame_path = os.path.join(feat_path, seg, 'resnet_frame_keras')
        images = os.listdir(seg_frame_path)
        images = get_N_data(feat_len_keep, images)
        video_x = []
        for image in images:
            seg_path = os.path.join(feat_path, seg, feat)
            image_path = os.path.join(seg_path, image)
            if not os.path.exists(image_path):
                video_x.append([0.0]*2048)
            else:
                with open(image_path) as fin:
                    feat_arr = list(map(float, fin.readline().split(',')))
                if len(feat_arr) < 2048:
                    feat_arr = feat_arr * 4
                    # feat_arr.extend([0.0]*(2048-len(feat_arr)))
                assert len(feat_arr) == 2048
                video_x.append(feat_arr)
        x.append(video_x)
        y.append(gt_dict[seg])
    return np.array(x), np.array(y)

# Get test features
def get_test_data(feat_path, gt_path, feat, feat_len_keep):
    with open(gt_path) as fin:
        test_list = []
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            test_list.append(line.split(' ')[0])
        test_list.sort()
    
    feat_list = []
    seg_list = []
    for seg in test_list:
        seg_frame_path = os.path.join(feat_path, seg, 'resnet_frame_keras')
        images = os.listdir(seg_frame_path)
        images = get_N_data(feat_len_keep, images)
        video_x = []
        for image in images:
            seg_path = os.path.join(feat_path, seg, feat)
            image_path = os.path.join(seg_path, image)
            if not os.path.exists(image_path):
                video_x.append([0.0]*2048)
            else:
                with open(image_path) as fin:
                    feat_arr = list(map(float, fin.readline().split(',')))
                if len(feat_arr) < 2048:
                    feat_arr = feat_arr * 4
                    # feat_arr.extend([0.0]*(2048-len(feat_arr)))
                assert len(feat_arr) == 2048
                video_x.append(feat_arr)
        feat_list.append(video_x)
        seg_list.append(seg)
    return np.array(feat_list), seg_list

# Get all input features
def get_multi_data(feat_path, gt_path, using_cues, feat_len_keep, func):
    input_list = []
    if 'face' in using_cues:
        x, help = func(feat_path, gt_path, 'arcface', feat_len_keep)
        x = x.reshape(-1, 2048)
        # x = preprocessing.MinMaxScaler().fit_transform(x)
        x = preprocessing.MaxAbsScaler().fit_transform(x)
        x = x.reshape(-1, feat_len_keep, 2048)
        print("face features' shape:", x.shape)
        input_list.append(x)
    for cue in ['head', 'upperbody', 'body', 'frame']:
        if cue in using_cues:
            x, help = func(feat_path, gt_path,'resnet_' + cue + '_keras', feat_len_keep)
            x = x.reshape(-1, 2048)
            # x = preprocessing.MinMaxScaler().fit_transform(x)
            x = preprocessing.MaxAbsScaler().fit_transform(x)
            x = x.reshape(-1, feat_len_keep, 2048)
            print(cue + " features' shape:", x.shape)
            input_list.append(x)
    return input_list, help
