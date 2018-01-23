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

def get_metadata_dataframe(tree_subset, no_tree_subset):
    """Returns dataframe with subimage metadata

    ARGUMENTS:
    - tree_subset (list): file names for subimages containing street trees
    - no_tree_subset (list): file names for subimages without street trees

    (dataframe): information for each subimage in the image
    """
    df = pd.DataFrame(index=range(0, len(tree_subset) + len(no_tree_subset)),
                      columns=('image_name', 'row', 'column', 'label',
                               'prediction', 'pred_round'))
    for i, item in enumerate(tree_subset):
        split_name = item.split('/')[-1].split('_')[:3]
        name = split_name[0]
        row = int(split_name[1])
        column = int(split_name[2])
        df.iloc[i] = [name, row, column, 1, None, None]
    for i, item in enumerate(no_tree_subset):
        split_name = item.split('/')[-1].split('_')[:3]
        name = split_name[0]
        row = int(split_name[1])
        column = int(split_name[2])
        df.iloc[len(tree_subset) + i] = [name, row, column, 0, None, None]
    return df

def get_label_visual(df, img):
    """Generates and saves a visual highlighting areas identified as containing
    street trees.

    ARGUMENTS:
    - df (dataframe)
    - img (array): image file
    """
    for row in range(50):
        for column in range(50):
            top = 100*row
            bottom = 100*(row + 1)
            left = 100*column
            right =100*(column+1)
            subimg = img[top:bottom,left:right,:]
            label = df[(df['row']==row) & (df['column']==column)].label.max()
            if label == 1:
                img[top:bottom,left:right,1] = \
                    (255-img[top:bottom,left:right,1])*.75 + \
                     img[top:bottom, left:right, 1]
    cv2.imwrite('labeled_visual.tif', img)

def get_prediction_visual(df, img, model, centers):
    """Generates and saves a visual highlighting areas identified as containing
    street trees.

    ARGUMENTS:
    - df (dataframe)
    - img (array): image file
    - model (Keras model): trained model to use for predictions
    - centers (array): RGB means of training data
    """
    for row in range(50):
        for column in range(50):
            top = 100*row
            bottom = 100*(row + 1)
            left = 100*column
            right = 100*(column+1)
            standardized_img = img[top:bottom,left:right,:] - centers
            prediction = model.predict(standardized_img.reshape(1, 100, 100, 3))[0, 0]
            index = df[(df['row']==row) & (df['column']==column)].index[0]
            df.iloc[index].prediction = prediction
            df.iloc[index].pred_round = round(prediction)
            if round(prediction) == 1:
                img[top:bottom,left:right,1] = \
                    (255-img[top:bottom,left:right,1])*.75 + \
                     img[top:bottom, left:right, 1]
    cv2.imwrite('predicted_visual.tif', img)
    return df

def main():
    conn, bucket = create_connection('treedata-ks')
    X_train, X_test, y_train, y_test = get_data(3, 141750)
    tree_subset, no_tree_subset = get_subimage_names(bucket, '10seg520820')
    metadata = get_metadata_dataframe(tree_subset, no_tree_subset)
    img = cv2.imread('trees_temp/10seg520820.tif', 1)
    get_label_visual(metadata, img)
    centers = np.mean(X_train, axis=(0, 1, 2))
    model = keras.models.load_model('trees_temp/final_model_141750images_13epochs_32batch_0.001lr_0.0reg_RGB_VGG_25dropout.h5')
    metadata = get_prediction_visual(metadata, img, model, centers)
    pickle.dump(metadata, open('trees_temp/visualization_metadata.p', "wb" ))

if __name__ == '__main__':
    main()
