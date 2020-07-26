import os
import random
import heapq
import numpy as np
from sklearn.metrics import accuracy_score
import argparse

# Get result
def get_result(seg_list, result):
    result_dict = {}
    for i in range(len(seg_list)):
        video_name = seg_list[i]
        if video_name not in result_dict:
            result_dict[video_name] = [result[i]]
        else:
            result_dict[video_name].append(result[i])
            
    output = []
    for video_name in result_dict:
        final_result = np.zeros(79, dtype=np.float32)
        for result in result_dict[video_name]:
            final_result += result
        final_result /= len(result_dict[video_name])
        output.append([video_name, final_result])
    return output

def calculate_accuracy(seg_list, result):
    y_true = [int(seg.split('_')[0]) for seg in seg_list]
    y_pred = [list(r).index(max(list(r))) for r in result]
    return accuracy_score(y_true, y_pred)

# Get mAP
def calculate_map(gt_val_path, my_val_path, top_k):
    id2videos = dict()
    with open(gt_val_path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            terms = line.strip().split(' ')
            id2videos[terms[0]] = terms[1:]
    id_num = len(lines)

    my_id2videos = dict()
    with open(my_val_path, 'r') as fin:
        lines = fin.readlines()
        assert(len(lines) <= id_num)
        for line in lines:
            terms = line.strip().split(' ')
            tmp_list = []
            for video in terms[1:]:
                if video not in tmp_list:
                    tmp_list.append(video)
            my_id2videos[terms[0]] = tmp_list

    ap_total = 0.
    for cid in id2videos:
        videos = id2videos[cid]
        if cid not in my_id2videos:
            continue
        my_videos = my_id2videos[cid][:top_k]
        # recall number upper bound
        # assert(len(my_videos) <= 100)
        ap = 0.
        ind = 0.
        for ind_video, my_video in enumerate(my_videos):
            if my_video in videos:
                ind += 1
                ap += ind / (ind_video + 1)
        ap_total += ap / len(videos)

    return ap_total / id_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get mAP')
    parser.add_argument('--ratio', default='4_3_3', help='training data ratio')
    parser.add_argument('--my_path', default='../log/mAP.log', help='result path')
    parser.add_argument('--top_k', type=int, default=1363, help='top k to calculate mAP') 
    args = parser.parse_args()

    test_gt_map_path = os.path.join('../data/CRV/gt', args.ratio, 'test_gt.txt')

    # print mAP
    print('mAP@' + str(args.top_k) + ':', calculate_map(test_gt_map_path, args.my_path, args.top_k))
