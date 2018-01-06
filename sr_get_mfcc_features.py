from scipy.io import wavfile
from python_speech_features import mfcc
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler


def pad_samples(sig, num_samples):
    return np.pad(sig, pad_width=(num_samples - len(sig), 0), mode='constant', constant_values=(0, 0))


def cut_samples(sig, num_samples, num):
    for i in range(num):
        beg = np.random.randint(0, len(sig) - num_samples)
        yield sig[beg: beg + num_samples]


def sr_get_mfcc_features(all_df, num_samples):
    sigs_all = []
    y_word_id = []
    y_word = []
    y_actual_word = []
    for row in tqdm(all_df.values):
        file = row[0]
        word_id = row[1]
        word = row[2]
        actual_word = row[3]
        rate, sig = wavfile.read(file)
        if len(sig) > num_samples:
            sigs = cut_samples(sig, num_samples, 1000)
            for s in sigs:
                sigs_all.append(s)
                y_word_id.append(word_id)
                y_word.append(word)
                y_actual_word.append(actual_word)
        else:
            if len(sig) < num_samples:
                sig = pad_samples(sig, num_samples)
            sigs_all.append(sig)
            y_word_id.append(word_id)
            y_word.append(word)
            y_actual_word.append(actual_word)
    x = np.zeros((len(y_word_id), 99, 13), dtype=np.float32)
    idx = 0
    for s in tqdm(sigs_all):
        mfcc_feat = mfcc(s)
        # if fbank_feat.shape != (99, 26):
        #     print(fbank_feat.shape, s.shape)
        x[idx, :, :] = StandardScaler().fit_transform(mfcc_feat)
        idx += 1
    lb = LabelBinarizer()
    lb.fit([i for i in range(12)])
    y = lb.transform(y_word_id)
    return x, y, y_word_id, y_word, y_actual_word
