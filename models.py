import os
from keras.layers import Input, LSTM, Reshape, Dense, AveragePooling1D, MaxPooling1D, Permute, multiply, GlobalAveragePooling1D, concatenate
from keras.models import Sequential, Model
from keras.layers.core import Lambda
import keras.backend
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

# Temporal models
class TemporalModels():
    def __init__(self, model, seq_length, layer_name, feature_length=2048):
        self.seq_length = seq_length
        self.layer_name = layer_name
        self.feature_length = feature_length
        
        if model == 'avg_pooling':
            print("Load avg pooling...")
            self.model = self.avg_pooling()
        elif model == 'max_pooling':
            print("Load max pooling...")
            self.model = self.max_pooling()
        elif model == 'lstm':
            print("Load lstm...")
            self.model = self.lstm()
        elif model == 'tam':
            print("Load tam...")
            self.model = self.tam()
        self.model.name = layer_name

    def avg_pooling(self):
        model = Sequential()
        model.add(AveragePooling1D(pool_size=self.seq_length, padding='valid'))
        # model.add(GlobalAveragePooling1D())
        model.add(Reshape((self.feature_length, 1)))
        return model
    
    def max_pooling(self):
        model = Sequential()
        model.add(MaxPooling1D(pool_size=self.seq_length, padding='valid'))
        model.add(Reshape((self.feature_length, 1)))
        return model
    
    def lstm(self):
        model = Sequential()
        model.add(LSTM(self.feature_length))
        model.add(Reshape((self.feature_length, 1)))
        return model
        
    def tam(self):
        x = Input(shape=(self.seq_length, self.feature_length))
        x1 = Permute((2, 1))(x)
        a = Dense(self.seq_length, activation='softmax')(x1) 
        a = Permute((2, 1), name='attention_vec')(a) 
        x2 = multiply([x, a])
        x2 = Lambda(lambda x: keras.backend.sum(x, axis=1))(x2) 
        x2 = Reshape((self.feature_length, 1))(x2)
        model = Model(inputs=x, outputs=x2)
        print(model.summary())
        return model
        
# Concatenate features
def ConcatFeature(using_cues, face_temporal, head_temporal, upperbody_temporal, body_temporal, frame_temporal):
    print("Using cues:", using_cues)
    if 'face' in using_cues:
        x = face_temporal
        if 'head' in using_cues: 
            x = concatenate([x, head_temporal])
        if 'upperbody' in using_cues:
            x = concatenate([x, upperbody_temporal])
        if 'body' in using_cues:
            x = concatenate([x, body_temporal])
        if 'frame' in using_cues:
            x = concatenate([x, frame_temporal])
    elif 'head' in using_cues: 
        x = head_temporal
        if 'upperbody' in using_cues:
            x = concatenate([x, upperbody_temporal])
        if 'body' in using_cues:
            x = concatenate([x, body_temporal])
        if 'frame' in using_cues:
            x = concatenate([x, frame_temporal])
    elif 'upperbody' in using_cues:
        x = upperbody_temporal
        if 'body' in using_cues:
            x = concatenate([x, body_temporal])
        if 'frame' in using_cues:
            x = concatenate([x, frame_temporal])
    elif 'body' in using_cues:
        x = body_temporal
        if 'frame' in using_cues:
            x = concatenate([x, frame_temporal])
    else:
        x = frame_temporal
    return x

# Get input layers
def InputLayerList(using_cues, face_input, head_input, upperbody_input, body_input, frame_input):
    input_list = []
    if 'face' in using_cues:
        input_list.append(face_input)
    if 'body' in using_cues:
        input_list.append(body_input)
    if 'head' in using_cues:
        input_list.append(head_input)
    if 'upperbody' in using_cues:
        input_list.append(upperbody_input)
    if 'frame' in using_cues:
        input_list.append(frame_input)
    return input_list
