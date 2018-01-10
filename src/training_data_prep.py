import os
import boto
from boto.s3.key import Key
from src.aws_functions import create_connection, get_bucket_contents
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def s3_image_to_array(bucket, filename):
    """Returns a grayscale numpy array from a filepath to an image on s3

    ARGUMENTS:
    - bucket (s3 bucket)
    - filename (string): s3 uri to tif file

    RETURNS:
    - numpy array
    """
    key = bucket.get_key(filename)
    key.get_contents_to_filename('trees_temp/tmp.tif')
    img = cv2.imread('trees_temp/tmp.tif', 0)
    os.remove('trees_temp/tmp.tif')
    return img

def get_image_array_lists(bucket):
    """Returns lists of arrays representing the labeled images.

    ARGUMENTS:
    - bucket (s3 bucket)

    RETURNS:
    - One list of image arrays, one list of labels (each array represents an
        image in greyscale)
    """
    data = []
    labels = []
    tree_files = get_bucket_contents(bucket, 'test_subset/HasStreetTree')
    no_tree_files = get_bucket_contents(bucket, 'test_subset/NoStreetTree')
    for f in tree_files:
        data.append(s3_image_to_array(bucket, f))
        labels.append(1)
    for f in no_tree_files:
        data.append(s3_image_to_array(bucket, f))
        labels.append(0)
    return data, labels

def split_training_data(data, labels):
    """Returns

    ARGUMENTS:
    - bucket (s3 bucket)
    - data (list of numpy arrays)
    - labels (list of binary values)
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
    return X_train, X_test, y_train, y_test

def save_data_to_s3(bucket, X_train, X_test, y_train, y_test):
    """Saves arrays to s3
    """
    np.savez('trees_temp/X_train.npz', *X_train)
    np.savez('trees_temp/X_test.npz', *X_test)
    np.save('trees_temp/y_train.npy', y_train)
    np.save('trees_temp/y_test.npy', y_test)
    for filename in os.listdir('trees_temp'):
        key = Key(bucket)
        key.key = 'test_train_data/' + filename
        key.set_contents_from_filename('trees_temp/' + filename)

def check_filepaths():
    if not os.path.exists('trees_temp'):
        os.makedirs('trees_temp')

def main():
    check_filepaths()
    conn, bucket = create_connection('treedata-ks')
    data, labels = get_image_array_lists(bucket)
    X_train, X_test, y_train, y_test = split_training_data(data, labels)
    save_data_to_s3(bucket, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
