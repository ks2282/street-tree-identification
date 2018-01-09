import fnmatch
from PIL import Image
import io, sys, os
import pandas as pd
import boto
from boto.s3.key import Key
from image_prep import ImageProcessor
from aws_functions import create_connection, get_bucket_contents, get_s3_image

def get_tree_data(destination, bucket):
    """Returns a dataframe with the street tree data.

    ARGUMENTS:
    - destination (string): local directory to store file
    - bucket (S3 bucket): where tree csv file lives

    RETURNS:
    - (dataframe): contains information in the street tree dataset
    """
    key = bucket.get_key('metadata/Street_Tree_List.csv')
    filename = destination + '/metadata/tree_data.csv'
    key.get_contents_to_filename(filename)
    df = pd.read_csv(filename)

    # Ignore trees without geocodes
    df = df[~pd.isnull(df['Location'])]

    # Convert planting dates to datetime
    df['PlantDate'] = pd.to_datetime(df['PlantDate'])

    # Exclude trees planted in or after April 2011 (imagery vintage)
    df = df[~((df['PlantDate'].dt.year == 2011) & (df['PlantDate'].dt.month > 3))]
    df = df[~(df['PlantDate'].dt.year > 2011)]

    return df

def get_imagery_metadata(destination, bucket):
    """Returns a dataframe with the street tree data.

    ARGUMENTS:
    - destination (string): local directory to store file
    - bucket (S3 bucket): where tree csv file lives

    RETURNS:
    - (dataframe): contains information in the street tree dataset
    """
    key = bucket.get_key('metadata/HIGH_RES_ORTHO_227914.txt')
    filename = destination + '/metadata/image_metadata.txt'
    key.get_contents_to_filename(filename)
    df = pd.read_csv(filename, usecols = ['Image Name',
                                 'NW Corner Lat dec', 'NW Corner Long dec',
                                 'SE Corner Lat dec', 'SE Corner Long dec',])

    return df

def process_s3_images(bucket, subfolder, image_metadata, output_path, side_length, tree_data):
    image_files = get_bucket_contents(bucket, subfolder)
    for filename in image_files:
        img = get_s3_image(bucket, filename)
        name = filename[len(subfolder)+1:-4] # assumes 3-character extension, such as .tif
        tile = ImageProcessor(img, image_metadata, name, output_path)
        tile.split_and_label(side_length, tree_data)

def load_to_s3(destination, bucket):
    """Loads labeled data to s3.

    ARGUMENTS:
    - bucket (s3 bucket)
    """
    for filename in os.listdir(destination + '/HasStreetTree'):
        key = Key(bucket)
        key.key = 'labeled_data/HasStreetTree/' + filename
        key.set_contents_from_filename(destination + '/HasStreetTree/' + filename)
    for filename in os.listdir(destination + '/NoStreetTree'):
        key = Key(bucket)
        key.key = 'labeled_data/NoStreetTree/' + filename
        key.set_contents_from_filename(destination + '/NoStreetTree/' + filename)

def check_filepaths():
    if not os.path.exists(destination + '/metadata'):
        os.makedirs(destination + '/metadata')
    if not os.path.exists(destination + '/HasStreetTree'):
        os.makedirs(destination + '/HasStreetTree')
    if not os.path.exists(destination + '/NoStreetTree'):
        os.makedirs(destination + '/NoStreetTree')

def main(destination, side_length):
    check_filepaths()
    conn, bucket = create_connection('treedata-ks')
    trees_df = get_tree_data(destination, bucket)
    img_metadata = get_imagery_metadata(destination, bucket)
    process_s3_images(bucket, 'originals', img_metadata, destination, side_length, trees_df)
    load_to_s3(destination, bucket)

if __name__ == '__main__':
    destination = sys.argv[1]
    side_length = int(sys.argv[2])
    main(destination, side_length)
