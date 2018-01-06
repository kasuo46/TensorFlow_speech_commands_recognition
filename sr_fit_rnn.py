from keras import optimizers, losses, activations, models
from keras.layers import GRU, Convolution2D, Dense, Input, Flatten, Dropout,\
    MaxPooling2D, BatchNormalization, Conv3D, ConvLSTM2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping


def sr_fit_rnn(x_train, y_train, x_val, y_val, x_test, y_test):
    model = Sequential()
    model.add(GRU(256, input_shape=(99, 13)))
    model.add(Dense(12, activation='softmax'))
    model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
    weights_path = 'data/model_weights_2.h5'
    callbacks_list = [ModelCheckpoint(filepath=weights_path, monitor='val_loss', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_categorical_accuracy', patience=5, verbose=1)]
    model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1, validation_data=(x_val, y_val),
              callbacks=callbacks_list)
    train_score = model.evaluate(x_train, y_train, batch_size=64)
    val_score = model.evaluate(x_val, y_val, batch_size=64)
    test_score = model.evaluate(x_test, y_test, batch_size=64)
    print(train_score, val_score, test_score)
    model.save('data/model_2.h5')
    return train_score, val_score, test_score
