import os, sys
import boto
from boto.s3.key import Key
from aws_functions import create_connection, get_bucket_contents
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def s3_image_to_array(bucket, filename, image_color_flag):
    """Returns a grayscale numpy array from a filepath to an image on s3

    ARGUMENTS:
    - bucket (s3 bucket)
    - filename (string): s3 uri to tif file
    - image_color_flag (integer): 1 for color, 0 for grayscale

    RETURNS:
    - numpy array
    """
    key = bucket.get_key(filename)
    key.get_contents_to_filename('trees_temp/tmp.tif')
    img = cv2.imread('trees_temp/tmp.tif', image_color_flag)
    os.remove('trees_temp/tmp.tif')
    return img

def get_image_array_lists(bucket, image_color_flag):
    """Returns lists of arrays representing the labeled images.

    ARGUMENTS:
    - bucket (s3 bucket)

    RETURNS:
    - One list of image arrays, one list of labels (each array represents an
        image in greyscale)
    """
    data = []
    labels = []
    tree_files = get_bucket_contents(bucket, 'labeled_data/HasStreetTree')
    no_tree_files = get_bucket_contents(bucket, 'labeled_data/NoStreetTree')
    for f in tree_files:
        img = s3_image_to_array(bucket, f, image_color_flag)
        if img.shape[0] == img.shape[1] == 100:
            data.append(img)
            labels.append(1)
    for f in no_tree_files:
        img = s3_image_to_array(bucket, f, image_color_flag)
        if img.shape[0] == img.shape[1] == 100:
            data.append(img)
            labels.append(0)
    return np.array(data), np.array(labels)

def split_training_data(data, labels):
    """Returns

    ARGUMENTS:
    - bucket (s3 bucket)
    - data (numpy array of numpy arrays)
    - labels (numpy array of binary values)
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
    return X_train, X_test, y_train, y_test

def save_data_to_s3(bucket, X_train, X_test, y_train, y_test, image_color_flag):
    """Saves arrays locally and to s3
    """
    if image_color_flag == 1:
        filename = 'test_train_data_color.npz'
    if image_color_flag == 0:
        filename = 'test_train_data_grayscale.npz'
    np.savez_compressed('trees_temp/' + filename,
                            X_train, X_test, y_train, y_test)
    key = Key(bucket)
    key.key = 'test_train_data/' + filename
    key.set_contents_from_filename('trees_temp/' + filename)

def check_filepaths():
    if not os.path.exists('trees_temp'):
        os.makedirs('trees_temp')

def main(image_color_flag):
    check_filepaths()
    conn, bucket = create_connection('treedata-ks')
    data, labels = get_image_array_lists(bucket, image_color_flag)
    X_train, X_test, y_train, y_test = split_training_data(data, labels)
    save_data_to_s3(bucket, X_train, X_test, y_train, y_test, image_color_flag)

if __name__ == '__main__':
    image_color_flag = int(sys.argv[1])
    main(image_color_flag)
