import os
import argparse
import random
import heapq
from keras.layers import Input, Flatten, Dense, BatchNormalization, Dropout, Permute, RepeatVector, multiply
from keras.layers.core import Lambda
from keras.models import Model
import keras.backend
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

import dataloader
import models
import get_mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--model', default='../output/model.hdf5', help='directory to load model')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--batch-size', type=int, default=16384, help='training batch size')
    parser.add_argument('--ratio', default='4_3_3', help='dataset partition ratio')
    parser.add_argument('--temporal-merge',default='avg_pooling', help='temporal modeling method (avg/max/lstm/tam)')
    parser.add_argument('--spatial-merge', default='none', help='multi-cue modeling method (none/weighted/mcam)')
    parser.add_argument('--weights', default='1,1,1,1,1', help='weights for concatenation')
    parser.add_argument('--frame-num', type=int, default=16, help='sampled frame number')
    parser.add_argument('--using-cues', default="face,upperbody,frame", help='using cues (face/head/upperbody/body/frame)')
    parser.add_argument('--input-mode', default='feature', help='input mode (image/feature)')
    parser.add_argument('--dataset', default='CRV', help='dataset')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--mlp', type=int, default=1, help='use mlp or not')
    parser.add_argument('--top-k', type=int, default=1363, help='keep top k values to calculate the mAP')
    args = parser.parse_args()

    random.seed(args.seed)
    using_cues = args.using_cues.split(',')
    
    # Choose dataset
    if args.dataset == 'CRV':
        feat_root_path = "../data/CRV/feat"
        test_gt_path = os.path.join('../data/CRV/gt', args.ratio, 'test.txt')
        test_gt_mAP_path = os.path.join('../data/CRV/gt', args.ratio, 'test_gt.txt')
        class_num = 79
    
    # Loading feature
    if args.input_mode == 'feature':
        print("Load test feature...")
        test_x, seg_list = dataloader.get_multi_data(feat_root_path, test_gt_path, using_cues, args.frame_num, dataloader.get_test_data)
    
    # input layers
    face_input = Input(shape=(args.frame_num, 2048), dtype='float32', name='face_input')
    head_input = Input(shape=(args.frame_num, 2048), dtype='float32', name='head_input')
    upperbody_input = Input(shape=(args.frame_num, 2048), dtype='float32', name='upperbody_input')
    body_input = Input(shape=(args.frame_num, 2048), dtype='float32', name='body_input')
    frame_input = Input(shape=(args.frame_num, 2048), dtype='float32', name='frame_input')
    
    # temporal modeling
    face_temporal = models.TemporalModels(args.temporal_merge, args.frame_num, 'face').model(face_input)
    head_temporal = models.TemporalModels(args.temporal_merge, args.frame_num, 'head').model(head_input)
    upperbody_temporal = models.TemporalModels(args.temporal_merge, args.frame_num, 'upperbody').model(upperbody_input)
    body_temporal = models.TemporalModels(args.temporal_merge, args.frame_num, 'body').model(body_input)
    frame_temporal = models.TemporalModels(args.temporal_merge, args.frame_num, 'frame').model(frame_input)
    
    # weights
    if args.spatial_merge == 'weighted':
        weights = [float(i) for i in args.weights.split(',')]
        assert len(weights) == 5
        face_temporal = Lambda(lambda x: weights[0]*x, name='weighted_face')(face_temporal)
        head_temporal = Lambda(lambda x: weights[1]*x, name='weighted_head')(head_temporal)
        upperbody_temporal = Lambda(lambda x: weights[2]*x, name='weighted_upperbody')(upperbody_temporal)
        body_temporal = Lambda(lambda x: weights[3]*x, name='weighted_body')(body_temporal)
        frame_temporal = Lambda(lambda x: weights[4]*x, name='weighted_frame')(frame_temporal)

    # concatenate features
    x = models.ConcatFeature(using_cues, face_temporal, head_temporal, upperbody_temporal, body_temporal, frame_temporal)
    feat_count = len(using_cues)

    # multi-cue modeling
    if args.spatial_merge == 'none' or args.spatial_merge == 'weighted':
        x = Permute((2, 1))(x)
        x = Flatten()(x)
    elif args.spatial_merge == 'mcam':
        a = Dense(feat_count, activation='softmax')(x) 
        a = Lambda(lambda x: keras.backend.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(2048)(a)
        a = Permute((2, 1), name='attention_vec')(a)
        x = Permute((2, 1))(x)
        x = multiply([x, a])
        x = Flatten()(x)

    # classification
    if args.mlp:
        x = Dropout(0.5)(x)
        x = Dense(2048, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
    output = Dense(class_num, activation='softmax')(x)
    
    # define, compile, and fit the model
    input = models.InputLayerList(using_cues, face_input, head_input, upperbody_input, body_input, frame_input)
    model = Model(inputs=input, output=output)
    
    # load weights
    model.load_weights(args.model)
    
    # predict result
    if args.input_mode == 'feature':
        print('Predict using features...')
        result = model.predict(x=test_x, batch_size=args.batch_size)
    output = list(zip(seg_list, result))
    
    # print accuracy
    print('accuracy:', get_mAP.calculate_accuracy(seg_list, result))

    # sort to get mAP
    print('Start sorting...')
    classification_result = [[] for i in range(class_num)]
    for i in range(class_num):
        top_k = heapq.nlargest(args.top_k, output, key = lambda x: x[1][i])
        classification_result[i] = top_k

    output_path = args.model.replace('hdf5', 'txt')
    with open(output_path, 'w') as fout:
        for i in range(class_num):
            output_str = str(i)
            top_k = classification_result[i]
            for data in top_k:
                # print("Data", data)
                output_str += (' ' + data[0])
            output_str += '\n'
            fout.write(output_str)
    
    # print mAP
    print('mAP@' + str(args.top_k) + ':', get_mAP.calculate_map(test_gt_mAP_path, output_path, args.top_k))
