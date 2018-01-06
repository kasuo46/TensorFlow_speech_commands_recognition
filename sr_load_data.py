import os
import pandas as pd


def sr_load_data(train_words_list, train_audio_dir, noise_dir, unknown_list):
    train_df_list = []
    for word_id, word in enumerate(train_words_list):
        for file in os.listdir(train_audio_dir + word + '/'):
            train_df_list.append([train_audio_dir + word + '/' + file, word_id, word, word])
    for file in os.listdir(noise_dir):
        if file.endswith('wav'):
            train_df_list.append([noise_dir + file, 10, 'silence', file[:-4]])
    for word in unknown_list:
        for file in os.listdir(train_audio_dir + word + '/'):
            train_df_list.append([train_audio_dir + word + '/' + file, 11, 'unknown', word])
    train_df = pd.DataFrame(train_df_list, columns=['file', 'word_id', 'word', 'actual_word'])
    return train_df
