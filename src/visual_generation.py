"""
Scripts for generating visuals for presentation
"""
import sys, boto, cv2, fnmatch, keras, pickle, numpy as np, pandas as pd
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

def get_label_visual(df, img, imagename):
    """Generates and saves a visual highlighting areas identified as containing
    street trees.

    ARGUMENTS:
    - df (dataframe)
    - img (array): image file
    """
    label_img = img.copy()
    for row in range(50):
        for column in range(50):
            top = 100*row
            bottom = 100*(row + 1)
            left = 100*column
            right =100*(column+1)
            label = df[(df['row']==row) & (df['column']==column)].label.max()
            if label == 1:
                label_img[top:bottom,left:right,1] = \
                    (255-label_img[top:bottom,left:right,1])*.75 + \
                     label_img[top:bottom, left:right, 1]
    cv2.imwrite(imagename + '_labeled_visual.tif', label_img)

def get_prediction_visual(df, img, model, centers, imagename):
    """Generates and saves a visual highlighting areas identified as containing
    street trees.

    ARGUMENTS:
    - df (dataframe)
    - img (array): image file
    - model (Keras model): trained model to use for predictions
    - centers (array): RGB means of training data
    """
    pred_img = img.copy()
    for row in range(50):
        for column in range(50):
            top = 100*row
            bottom = 100*(row + 1)
            left = 100*column
            right = 100*(column+1)
            standardized_img = pred_img[top:bottom,left:right,:] - centers
            prediction = model.predict(standardized_img.reshape(1, 100, 100, 3))[0, 0]
            index = df[(df['row']==row) & (df['column']==column)].index
            df.prediction.iloc[index] = prediction
            df.pred_round.iloc[index] = round(prediction)
            if round(prediction) == 1:
                pred_img[top:bottom,left:right,1] = \
                    (255-pred_img[top:bottom,left:right,1])*.75 + \
                     pred_img[top:bottom, left:right, 1]
    cv2.imwrite(imagename + '_predicted_visual.tif', pred_img)
    return df

def main(imagename):
    conn, bucket = create_connection('treedata-ks')
    X_train, X_test, y_train, y_test = get_data(3, 141750)
    tree_subset, no_tree_subset = get_subimage_names(bucket, imagename)
    metadata = get_metadata_dataframe(tree_subset, no_tree_subset)
    print(metadata.shape)
    img = cv2.imread('trees_temp/' + imagename + '.tif', 1)
    get_label_visual(metadata, img, imagename)
    centers = np.mean(X_train, axis=(0, 1, 2))
    model = keras.models.load_model('trees_temp/final_model_141750images_13epochs_32batch_0.001lr_0.0reg_RGB_VGG_25dropout.h5')
    metadata_pred = get_prediction_visual(metadata, img, model, centers, imagename)
    print(metadata_pred.shape)
    pickle.dump(metadata_pred, open('trees_temp/' + imagename + '_visualization_metadata.p', "wb" ))

if __name__ == '__main__':
    imagename = sys.argv[1]
    main(imagename)
