import fnmatch
import io, sys, os
import pandas as pd
import boto
from boto.s3.key import Key
from aws_functions import create_connection, get_bucket_contents, get_s3_image



def main():
    conn, bucket = create_connection('treedata-ks')

if __name__ == '__main__':
    destination = sys.argv[1]
    side_length = int(sys.argv[2])
    main(destination, side_length)
