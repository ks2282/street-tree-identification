"""
The ImageProcessor class below is used to split an image into subimages, and
saves each subimage in a directory depending on it's labeled class.
ImageProcessor is imported in the aws_image_processing.py script for splitting
and labeling all images contained in a specified S3 bucket.
"""
from PIL import Image
import pandas as pd
import numpy as np
import os

def get_centroid(coordinates):
    """Returns centroid of coordinates.

    ARGUMENTS:
    - coordinates (sequence of tuples): each tuple contains (latitude, longitude)

    RETURNS:
    - (latitude, longitude)
    """
    return tuple([np.mean(coords) for coords in zip(*coordinates)])

class ImageProcessor(object):
    """ Splits and labels an image for street tree content
    """
    def __init__(self, img, image_metadata, filename, output_path):
        """
        ARGUMENTS:
        - img (Image): image to process
        - image_metadata (dataframe): metadata file containing name and bounding
            coordinates of each subimage
        """
        self.img = img # the image
        self.width = self.img.size[0] # width of the image
        self.length = self.img.size[1] # length of the image
        self.output_path = output_path # directory to write subimages to
        self.image_metadata = image_metadata # image metadata file
        self.filename = filename # name of file without extension
        self.ext = '.tif' # file extension

        self.image_NW, self.image_SE = self._get_image_coordinates() # bounding coordinates

    def _get_image_coordinates(self):
        """Returns bounding coordinates of the image.

        RETURNS:
        - image_NW (float, float): northwest corner coordinates (latitude, longitude)
        - image_SE (float, float): southeast corner coordinates (latitude, longitude)
        """
        name = self.filename.lower()
        record = self.image_metadata[self.image_metadata['Image Name'] == name]
        image_NW = (record['NW Corner Lat dec'].iloc[0],
                    record['NW Corner Long dec'].iloc[0])
        image_SE = (record['SE Corner Lat dec'].iloc[0],
                    record['SE Corner Long dec'].iloc[0])
        return image_NW, image_SE

    def _get_subimage(self, row, column, side_length):
        """Returns a subimage an image for a specified row and column.

        ARGUMENTS:
        - row (int)
        - column (int)
        - side_length (int)

        RETURNS:
        - subimage (Image)
        """
        bbox = (column*side_length,
                row*side_length,
                column*side_length + side_length,
                row*side_length + side_length)
        subimage = self.img.crop(bbox)

        return subimage

    def _get_subimage_coordinates(self, row, column, coord_step):
        """Returns bounding coordinates for the subimage.

        ARGUMENTS:
        - row (int)
        - column (int)
        - coord_step (tuple): coordinate offset per subimage step

        RETURNS:
        - subimage_NW (tuple of floats): subimage northwest coordinates
        - subimage_SE (tuple of floats): subimage southeast coordinates
        """
        subimage_NW = (self.image_NW[0] + row*coord_step[0],
                       self.image_NW[1] + column*coord_step[1])
        subimage_SE = (self.image_NW[0] + (row+1)*coord_step[0],
                       self.image_NW[1] + (column+1)*coord_step[1])
        return subimage_NW, subimage_SE

    def _contains_tree(self, tree_data, subimage_NW, subimage_SE):
        """Returns whether the subimage contains a tree.

        ARGUMENTS:
        - tree_data (dataframe): contains coordinates for each known street tree
        - subimage_NW (tuple of floats)
        - subimage_SE (tuple of floats)

        RETURNS:
        - boolean
        """
        included_tree = tree_data.copy()
        included_tree = included_tree[included_tree['Latitude'] < subimage_NW[0]]
        included_tree = included_tree[included_tree['Latitude'] > subimage_SE[0]]
        included_tree = included_tree[included_tree['Longitude'] > subimage_NW[1]]
        included_tree = included_tree[included_tree['Longitude'] < subimage_SE[1]]
        if included_tree.shape[0] > 0:
            return True
        return False

    def split_and_label(self, side_length, tree_data):
        """
        ARGUMENTS:
        - side_length (int): number of bits, defining side length of small
            square-shaped subimage
        - tree_data (dataframe): contains coordinates for each known street tree
        """
        coord_step = tuple([(coords[0] - coords[1])/(self.width//side_length)
                            for coords in zip(self.image_SE, self.image_NW)])
        for column in range(self.width//side_length):
            for row in range(self.length//side_length):
                subimage = self._get_subimage(row, column, side_length)
                subimage_NW, subimage_SE = self._get_subimage_coordinates(row, column, coord_step)
                if self._contains_tree(tree_data, subimage_NW, subimage_SE):
                    path = self.output_path + '/HasStreetTree/'
                else:
                    path = self.output_path + '/NoStreetTree/'
                subimage_centroid = get_centroid((subimage_NW, subimage_SE))
                new_filename = self.filename + '_' + str(row) + '_' + str(column) \
                                + '_' + str(subimage_centroid[0]) + '_' + \
                                str(subimage_centroid[1]) + self.ext
                subimage.save(path + new_filename)
