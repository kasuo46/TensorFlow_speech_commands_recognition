import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from python_speech_features import mfcc
from python_speech_features import logfbank
# from sr_load_data import *
# from sr_get_fbank_features import *
# from sr_get_val_test_index import *
# from sklearn.preprocessing import StandardScaler
#
# train_audio_dir = 'train/audio/'
# noise_dir = 'train/audio/_background_noise_/'
# train_words_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
# noise_list = ['_background_noise_']
# unknown_list = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
#                 'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow']
# NUM_SAMPLES = 16000
#
# all_df = sr_load_data(train_words_list, train_audio_dir, noise_dir, unknown_list)
#
# train_index, test_index, val_index = sr_get_val_test_index(all_df)

# test_df = all_df.loc[test_index]
# # print(test_df.head())
#
# val_df = all_df.loc[val_index]
# print(val_df.head())

# for file in tqdm(train_df['file'].values):
#     rate, sig = wavfile.read(file)
#     if len(sig) > 16000:
#         print(file)
#         print(rate, sig.shape)

#
# def chop_audio(sample1, length=6000, num=10):
#     for i in range(num):
#         beg = np.random.randint(0, len(sample1) - length)
#         yield sample1[beg: beg + length]
#
#
# a = chop_audio(sig)
# for x in a:
#     print(x.shape)
# wavfile.write('white_noise.wav', 16000,
#  np.array(((acoustics.generator.noise(16000*60, color='white'))/3) * 32767).astype(np.int16))
# rate, sig = wavfile.read('train/audio/_background_noise_/dude_miaowing_1.wav')

# rate, sig = wavfile.read('train/audio/bed/0a7c2a8d_nohash_0.wav')
# filter_banks = logfbank(sig)
# print(filter_banks.shape)
# print(np.mean(filter_banks, axis=0))

# test_list = pd.read_table('train/testing_list.txt', sep='\n', header=None, names=['file'])
# test_list['file'] = 'train/audio/' + test_list['file']
# print(test_list.shape, test_list.columns)
# # print(test_list.head())
# val_list = pd.read_table('train/validation_list.txt', sep='\n', header=None, names=['file'])
# val_list['file'] = 'train/audio/' + val_list['file']
# print(val_list.shape, val_list.columns)
# # print(val_list.head())
# # print(train_df.head())
# test_merge = pd.merge(train_df, test_list, on='file', how='inner', right_index=True)
# print(test_merge.shape)
# print(list(test_merge.index.values))
# # print(test_merge.head(100))
# # test_merge.to_csv('test_merge.csv')
# val_merge = pd.merge(train_df, val_list, on='file', how='inner', right_index=True)
# print(val_merge.shape)
# print(list(val_merge.index.values))
# # print(val_merge.head(100))

# x_val, y_val, y_val_word_id, y_val_word, y_val_actual_word = sr_get_fbank_features(all_df.loc[val_index],
#                                                                                    NUM_SAMPLES)
#
# scaler = StandardScaler()
# print(scaler.fit(x_val[0]))
# print(scaler.mean_)
# print(scaler.transform(x_val[0]))
from keras import optimizers, losses, activations, models
from keras.layers import GRU, Convolution2D, Dense, Input, Flatten, Dropout,\
    MaxPooling2D, BatchNormalization, Conv3D, ConvLSTM2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping


model = Sequential()
model.add(GRU(512, input_shape=(99, 26), return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(256, return_sequences=True))
model.add(Dropout(0.5))
model.add(GRU(128))
model.add(Dropout(0.5))
model.add(Dense(12, activation='softmax'))
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()
