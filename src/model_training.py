import fnmatch
import io, sys, os
import pandas as pd
import boto
from boto.s3.key import Key
from aws_functions import create_connection, get_bucket_contents

def get_s3_image(bucket, filename):
    """Returns an image from a filepath to S3

    ARGUMENTS:
    - filepath: string, s3 uri to tif file

    RETURNS:
    - Image
    """
    key = bucket.get_key(filename)
    tmp = io.BytesIO()
    key.get_contents_to_file(tmp)
    return Image.open(tmp)

def main():
    conn, bucket = create_connection('treedata-ks')
    print(bucket)

if __name__ == '__main__':
    #destination = sys.argv[1]
    #side_length = int(sys.argv[2])
    main()
