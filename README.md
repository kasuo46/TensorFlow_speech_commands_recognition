# Speech Commands Recognition
This project is to build a RNN model on the speech commands dataset released by TensorFlow. The mfcc and filter bank features are extracted from the audio files as the input features. The RNN models contains GRU and Dense layers in it. This model can achieve the accuracies of 98.5% on training, 94.8% on validation and 94.9% on testing.

## Introduction to the Dataset
TensorFlow recently released the Speech Commands Datasets. It includes 65,000 one-second long utterances of 30 short words, by thousands of different people. Twenty core command words were recorded, with most speakers saying each of them five times. The core words are "Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go", "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", and "Nine". To help distinguish unrecognized words, there are also ten auxiliary words, which most speakers only said once. These include "Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", and "Wow". There is also background_noise contains longer clips of "silence" that you can break up and use as training input.

The dataset is available here https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html.

## Libraries Used in This Project
* python_speech_features: http://python-speech-features.readthedocs.io/en/latest/
* Signal from SciPy

## Functions of Each File
* `speech_recognition_EDA.ipynb` is the EDA of the dataset.
* `sr_load_data.py` loads the input data and generate a pandas DataFrame contains the file paths, words, word ids, categories.
* `sr_get_train_val_test_index.py` separates the data into training, validation and testing.
* `sr_get_mfcc_features.py` extracts the mfcc features for all data.
* `sr_get_fbank_features.py` extracts the mfcc features for all data.
* `sr_fit_rnn.py` builds the sequential model in Keras, fits and validates the performance of the model.
* `main_v2.py` is the main function.

## Possible Further Improvements
* Consider using deep CNN to train.
* Reduce the sampling rate to make it more computationally efficient.
* Preprocess the audio files to reduce the noise.

## References
Thanks to the authors below who provide excellent kernels:

https://www.kaggle.com/davids1992/speech-representation-and-data-exploration

https://www.kaggle.com/alexozerin/end-to-end-baseline-tf-estimator-lb-0-72

https://www.kaggle.com/mindcool/lb-0-77-keras-gru-with-filter-banks-features
