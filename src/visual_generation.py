"""
Scripts for generating visuals for presentation
"""
import boto
from boto.s3.key import Key
from aws_functions import create_connection, get_bucket_contents
import cv2
import numpy as np
import fnmatch
import pandas as pd
import keras

def get_subimage_names(bucket, imagename):
    """Gets labeled subimage filenames from S3.

    ARGUMENTS:
    - imagename (string)
    """
    tree_files = get_bucket_contents(bucket, 'labeled_data/HasStreetTree')
    no_tree_files = get_bucket_contents(bucket, 'labeled_data/NoStreetTree')

    tree_files_subset = fnmatch.filter(tree_files, \
                                       'labeled_data/HasStreetTree/' + \
                                       imagename + '*')
    no_tree_files_subset = fnmatch.filter(tree_files, \
                                          'labeled_data/NoStreetTree/' + \
                                          imagename + '*')
