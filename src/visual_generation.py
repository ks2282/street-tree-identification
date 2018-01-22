"""
Scripts for generating visuals for presentation
"""
import boto, cv2, fnmatch, keras, numpy as np, pandas as pd
from boto.s3.key import Key
from aws_functions import create_connection, get_bucket_contents
from model_training import get_data

def get_subimage_names(bucket, imagename):
    """Gets labeled subimage filenames from S3.

    ARGUMENTS:
    - imagename (string)

    RETURNS:
    - (list, list): (image paths with trees, image paths without trees)
    """
    tree_files = get_bucket_contents(bucket, 'labeled_data/HasStreetTree')
    no_tree_files = get_bucket_contents(bucket, 'labeled_data/NoStreetTree')

    tree_files_subset = fnmatch.filter(tree_files, \
                                       'labeled_data/HasStreetTree/' + \
                                       imagename + '*')
    no_tree_files_subset = fnmatch.filter(tree_files, \
                                          'labeled_data/NoStreetTree/' + \
                                          imagename + '*')
    return tree_files_subset, no_tree_files_subset

def get_training_mean(X):
    """
    """
    return np.mean(X, axis=(0, 1, 2))

def get_label_visual(subimage_info, img):
    """Generates and saves a visual highlighting areas identified as containing
    street trees.

    ARGUMENTS:
    - subimage_info (dataframe)
    - img (array): image file
    """
    for row in range(50):
        for column in range(50):
            top = 100*row
            bottom = 100*(row + 1)
            left = 100*column
            right =100*(column+1)
            subimg = img[top:bottom,left:right,:]
            label = tree_sub_df[(tree_sub_df['row']==row) & \
                                (tree_sub_df['column']==column)].label.max()
            if label == 1:
                img[top:bottom,left:right,1] = \
                    (255-img[top:bottom,left:right,1])*.75 + \
                     img[top:bottom, left:right, 1]
    cv2.imwrite('labeled_visual.tif', img)

def get_metadata_dataframe(subimage_name):
    """Downloads necessary subimages from S3.
    """

def load_model(model_filepath):
    """Loads model for prediction.
    """

def predict_subimage(subimage):
    """
    """


def main():
    X_train, X_test, y_train, y_test = get_data(3, 141750)
    centers = get_training_mean(X_train)


    y_pred = predict_test(final_model, X_test, y_test)
    save_data(final_model, X_test, y_test, y_pred)

if __name__ == '__main__':
    main()
