from PIL import Image
import pandas as pd
import os

class ImageTile(object):
    """

    """

    def __init__(self, image_path, image_metadata):
        """
        INPUTS:
        - image_path (string): path to access image file
        - image_metadata (dataframe)
        """
        self.img = Image.open(image_path)
        self.width = self.img.size[0]
        self.length = self.img.size[1]
        self.dirname = os.path.dirname(image_path)
        self.image_metadata = image_metadata

        base = os.path.basename(image_path)
        self.filename = os.path.splitext(base)[0]
        self.ext = os.path.splitext(base)[1]

        self.tile_NW, self.tile_SE = self._get_tile_coordinates()
        self.coord_step = None
        self.side_length = None

    def _get_tile_coordinates(self):
        """

        """
        name = self.filename.lower()
        record = self.image_metadata[self.image_metadata['Image Name'] == name]
        tile_NW = (record['NW Corner Lat dec'].iloc[0],
                    record['NW Corner Long dec'].iloc[0])
        tile_SE = (record['SE Corner Lat dec'].iloc[0],
                    record['SE Corner Long dec'].iloc[0])
        return (tile_NW, tile_SE)

    def crop_tile(self, row, column):
        """
        Crops an image tile for a specified portion of the image

        INPUTS:
        - row

        OUTPUTS:
        - slice_bit (Image)
        """
        bbox = (column*self.side_length,
                row*self.side_length,
                column*self.side_length + self.side_length,
                row*self.side_length + self.side_length)
        split = self.img.crop(bbox)

        self.coord_step = tuple([(coords[0] - coords[1])/(self.width//self.side_length)
                                    for coords in zip(self.tile_SE, self.tile_NW)])
        split_NW = (self.tile_NW[0] + row*self.coord_step[0],
                    self.tile_NW[1] + column*self.coord_step[1])
        split_SE = (self.tile_NW[0] + (row+1)*self.coord_step[0],
                    self.tile_NW[1] + (column+1)*self.coord_step[1])
        return split, split_NW, split_SE

    def _classify_image(self, row, column, tree_data, split_NW, split_SE):
        """
        Crops an image tile for a specified portion of the image

        INPUTS:
        - row (int)
        - column (int)
        - tree_data (dataframe)

        OUTPUTS:
        - (boolean)
        """
        trees = list(zip(tree_data.Latitude, tree_data.Longitude))
        for tree in trees:
            latcheck = (tree[0] < split_NW[0]) & (tree[0] > split_SE[0])
            longcheck = (tree[1] > split_NW[1]) & (tree[1] < split_SE[1])
            if latcheck & longcheck == True:
                return True
        return False

    def split_and_classify(self, side_length, tree_data):
        """
        INPUTS:``
        - side_length (int): number of bits, defining side length of small
            square-shaped tile segment
        - tree_data (dataframe)

        OUTPUTS:
        - None
        """
        self.side_length = side_length
        for column in range(self.width//self.side_length):
            for row in range(self.length//self.side_length):
                split, split_NW, split_SE = self.crop_tile(row, column)
                if self._classify_image(row, column, tree_data, split_NW, split_SE):
                    path = self.dirname + '/HasStreetTree/'
                else:
                    path = self.dirname + '/NoStreetTree/'
                new_filename = self.filename + '_' + str(row) + '_' + str(column) + self.ext
                split.save(path + new_filename)
