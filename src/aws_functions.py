import boto
import fnmatch

def create_connection(bucket_name):
    """Returns s3 connection and specified bucket.

    ARGUMENTS:
    - bucket_name (string): name of bucket to retrieve

    RETURNS:
    (connection, bucket): S3 connection, S3 bucket
    """
    conn = boto.connect_s3()
    bucket = conn.get_bucket(bucket_name)
    return conn, bucket


def get_bucket_contents(bucket, subfolder):
    """Returns content from an S3 bucket for a specified subfolder

    ARGUMENTS:
    - bucket (S3 bucket)
    - subfolder (string)

    RETURNS:
    (list): files contained in specified subfolder
    """
    lst = [key.name for key in bucket]
    return fnmatch.filter(lst, subfolder + '/*')[1:]
