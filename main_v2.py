import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
from python_speech_features import mfcc
from python_speech_features import logfbank
from sr_load_data import *
from sr_get_mfcc_features import *
from sr_get_train_val_test_index import *
from sr_fit_rnn import *

train_audio_dir = '/Users/Alex/PycharmProjects/speech_recognition_v1/train/audio/'
noise_dir = '/Users/Alex/PycharmProjects/speech_recognition_v1/train/audio/_background_noise_/'
train_words_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
noise_list = ['_background_noise_']
unknown_list = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow']
NUM_SAMPLES = 16000

all_df = sr_load_data(train_words_list, train_audio_dir, noise_dir, unknown_list)

train_index, test_index, val_index = sr_get_train_val_test_index(all_df)

x_train, y_train, y_train_word_id, y_train_word, y_train_actual_word = sr_get_mfcc_features(all_df.loc[train_index],
                                                                                             NUM_SAMPLES)
# print(x_train.shape, y_train.shape, len(y_train_word_id), len(y_train_word), len(y_train_actual_word))

x_test, y_test, y_test_word_id, y_test_word, y_test_actual_word = sr_get_mfcc_features(all_df.loc[test_index],
                                                                                        NUM_SAMPLES)
# print(x_test.shape, y_test.shape, len(y_test_word_id), len(y_test_word), len(y_test_actual_word))

x_val, y_val, y_val_word_id, y_val_word, y_val_actual_word = sr_get_mfcc_features(all_df.loc[val_index],
                                                                                   NUM_SAMPLES)
# print(x_val.shape, y_val.shape, len(y_val_word_id), len(y_val_word), len(y_val_actual_word))

train_score, val_score, test_score = sr_fit_rnn(x_train, y_train, x_val, y_val, x_test, y_test)

