import os, sys
import boto
from boto.s3.key import Key
from src.aws_functions import create_connection
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def split_validation_data(X, y):
    """Returns training and validation sets

    ARGUMENTS:
    - X (numpy array)
    - y (numpy array)
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
    return X_train, X_val, y_train, y_val

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
    key = bucket.get_key('test_subset/' + filename)
    key.get_contents_to_filename('trees_temp/' + filename)

def get_data(image_color_flag):
    """Checks if data is local, downloads if not

    ARGUMENTS:
    - image_color_flag (int): specifies whether to find grayscale (0) or color (1)
    """
    if image_color_flag == 0:
        filename = 'test_train_data.npz'
    if image_color_flag == 1:
        filename = 'test_train_data_color.npz'
    if not os.path.exists('trees_temp/' + filename):
        download_s3_data(filename)
    X_train, X_test, y_train, y_test = restore_matrices('trees_temp/' + filename)
    return X_train, X_test, y_train, y_test

#def nn_model():

def check_filepaths():
    if not os.path.exists('trees_temp'):
        os.makedirs('trees_temp')

def main(image_color_flag):
    check_filepaths()
    X_train, X_test, y_train, y_test = get_data(image_color_flag)
    X_train, X_val, y_train, y_val = split_validation_data(X_train, y_train)

if __name__ == '__main__':
    image_color_flag = int(sys.argv[1])
    main(image_color_flag)
