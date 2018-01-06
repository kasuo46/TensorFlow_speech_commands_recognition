import pandas as pd


def sr_get_train_val_test_index(train_df):
    test_list = pd.read_table('/Users/Alex/PycharmProjects/speech_recognition_v1/train/testing_list.txt',
                              sep='\n', header=None, names=['file'])
    test_list['file'] = '/Users/Alex/PycharmProjects/speech_recognition_v1/train/audio/' + test_list['file']
    val_list = pd.read_table('/Users/Alex/PycharmProjects/speech_recognition_v1/train/validation_list.txt', sep='\n', header=None, names=['file'])
    val_list['file'] = '/Users/Alex/PycharmProjects/speech_recognition_v1/train/audio/' + val_list['file']
    test_merge = pd.merge(train_df, test_list, on='file', how='inner', right_index=True)
    val_merge = pd.merge(train_df, val_list, on='file', how='inner', right_index=True)
    test_set = set(test_merge.index.values)
    val_set = set(val_merge.index.values)
    train_set = set(train_df.index.values)
    train_list = list(train_set.difference(test_set).difference(val_set))
    return train_list, list(test_merge.index.values), list(val_merge.index.values)
