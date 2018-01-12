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
from keras.applications.vgg16 import VGG16

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
        print('Downloading data from S3.')
        download_s3_data(filename)
    X_train, X_test, y_train, y_test = restore_matrices('trees_temp/' + filename)
    X_train = X_train[:training_size]
    y_train = y_train[:training_size]
    if image_color_flag == 0:
        X_train = X_train.reshape(X_train.shape[0], 100, 100, 1)
    if image_color_flag == 1:
        X_train = X_train.reshape(X_train.shape[0], 100, 100, 3)
    y_train = y_train.reshape(y_train.shape[0], 1)
    return X_train, y_train

def train_val_split(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    return X_train, X_val, y_train, y_val

def precision_recall(X, y, model):
    """Return precision and recall for the feature and label data.

    ARGUMENTS:
    X (numpy array)
    y (numpy array)
    """
    y_pred = model.predict(X).round()
    P = float(np.sum(y == 1))
    TP = float(np.sum((y_pred == 1) & (y == 1)))
    FP = float(np.sum((y_pred == 1) & (y == 0)))
    if np.sum(y_pred == 1) > 0:
        precision = TP/(TP + FP)
    else: precision = 'No predicted positives.'
    if P > 0:
        recall = TP/P
    else: recall = 'No positive labels in validation set.'
    return precision, recall

def nn_model(X_train, X_val, y_train, y_val, num_epochs, batch_size, image_color_flag, learning_rate):
    if image_color_flag == 0:
        input_shape = (100, 100, 1)
    else: input_shape = (100, 100, 3)
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
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(lr=learning_rate),
              metrics=['accuracy'])

    model.fit(X_train, y_train,
          verbose=1,
          batch_size = batch_size,
          epochs=num_epochsy)

    score = model.evaluate(X_val, y_val, verbose=0)
    print('Validation loss:' , score[0])
    print('Validation accuracy: ', score[1])

    precision, recall = precision_recall(X_val, y_val, model)
    print('Validation precision: ', precision)
    print('Validation recall: ', recall)

    return model

def vgg_model(X_train, X_val, y_train, y_val, num_epochs, batch_size, image_color_flag, learning_rate):
    if image_color_flag == 0:
        input_shape = (100, 100, 1)
    else: input_shape = (100, 100, 3)
    X_train = X_train.astype('float32')
    X_train /= 255

    model = VGG16(weights=None, input_shape=input_shape, classes=1)

    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(lr=learning_rate),
              metrics=['accuracy'])

    model.fit(X_train, y_train,
          batch_size = batch_size,
          epochs = num_epochs,
          verbose = 1)

    score = model.evaluate(X_val, y_val, verbose=0)
    print('Validation loss:' , score[0])
    print('Validation accuracy: ', score[1])

    precision, recall = precision_recall(X_val, y_val, model)
    print('Validation precision: ', precision)
    print('Validation recall: ', recall)

    return model


def check_filepaths():
    if not os.path.exists('trees_temp'):
        os.makedirs('trees_temp')

def main(image_color_flag, training_size, num_epochs, batch_size, learning_rate, vgg):
    check_filepaths()
    filename = 'trees_temp/kerasmodel_' \
                + str(training_size) + 'images_' \
                + str(num_epochs) + 'epochs_' \
                + str(batch_size) + 'batch_' \
                + str(learning_rate) + 'lr'
    if image_color_flag == 1: filename += '_RGB'
    X_train, y_train = get_data(image_color_flag, training_size)
    X_train, X_val, y_train, y_val = train_val_split(X_train, y_train)
    if vgg == 1:
        model = vgg_model(X_train, X_val, y_train, y_val, num_epochs, batch_size, \
                            image_color_flag, learning_rate)
        filename += '_vgg'
    else:
        model = nn_model(X_train, X_val, y_train, y_val, num_epochs, batch_size, \
                            image_color_flag, learning_rate)
    model.save(filename + '.h5')

if __name__ == '__main__':
    image_color_flag = int(sys.argv[1])
    training_size = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    vgg = int(sys.argv[6])
    main(image_color_flag, training_size, num_epochs, batch_size, learning_rate, vgg)
