def create_connection(bucket_name):
    """Returns s3 connection and specified bucket.

    ARGUMENTS:
    - bucket_name (string)
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
    - list of files contained in specified subfolder
    """
    lst = [key.name for key in bucket]
    return fnmatch.filter(lst, subfolder + '/*')[1:]

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
