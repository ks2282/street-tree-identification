import os, sys
import boto
from boto.s3.key import Key
from aws_functions import create_connection
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

def restore_matrices(npz_filepath):
    """Returns training and test data as numpy arrays

    ARGUMENTS:
    npz_filepath (string)
    """
    X_train = np.load(npz_filepath)['arr_0']
    X_test = np.load(npz_filepath)['arr_1']
    y_train = np.load(npz_filepath)['arr_2']
    y_test = np.load(npz_filepath)['arr_3']
    return X_train, X_test, y_train, y_test

def download_s3_data(filename):
    """Downloads npz file from s3

    ARGUMENTS:
    - filename
    """
    conn, bucket = create_connection('treedata-ks')
    key = bucket.get_key('test_train_data/' + filename)
    key.get_contents_to_filename('trees_temp/' + filename)

def get_data(image_color_flag, training_size):
    """Checks if data is local, downloads if not

    ARGUMENTS:
    - image_color_flag (int): specifies whether to find grayscale (0) or color (1)
    - training_size (int): limits amount of data to train
    """
    if image_color_flag == 0:
        filename = 'test_train_data.npz'
    if image_color_flag == 1:
        filename = 'test_train_data_color.npz'
    if not os.path.exists('trees_temp/' + filename):
        download_s3_data(filename)
    X_train, X_test, y_train, y_test = restore_matrices('trees_temp/' + filename)
    X_train = X_train[:training_size]
    y_train = y_train[:training_size]
    X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)
    y_train = keras.utils.to_categorical(y_train, 0)
    return X_train, y_train

def nn_model(X_train, y_train):
    input_shape = (100, 100, 1)
    X_train = X_train.astype('float32')
    X_train /= 255

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

    model.fit(X_train, y_train,
          verbose=1,
          epochs=10,
          validation_split=0.3)

    score = model.evaluate(X_train, y_train, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def check_filepaths():
    if not os.path.exists('trees_temp'):
        os.makedirs('trees_temp')

def main(image_color_flag, training_size):
    check_filepaths()
    X_train, y_train = get_data(image_color_flag, training_size)
    nn_model(X_train, y_train)

if __name__ == '__main__':
    image_color_flag = int(sys.argv[1])
    training_size = int(sys.argv[2])
    main(image_color_flag, training_size)
