"""
Validation data has been commented out as model has been selected.
"""
import os, sys, boto, pickle, numpy as np
from boto.s3.key import Key
from aws_functions import create_connection
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.initializers import Constant
from keras.regularizers import l2

def restore_matrices(npz_filepath):
    """Returns training and test data as numpy arrays.

    ARGUMENTS:
    npz_filepath (string)

    RETURNS:
    (array, array, array, array): (training features, testing features,
                                   training labels, testing labels)
    """
    X_train = np.load(npz_filepath)['arr_0']
    X_test = np.load(npz_filepath)['arr_1']
    y_train = np.load(npz_filepath)['arr_2']
    y_test = np.load(npz_filepath)['arr_3']
    return X_train, X_test, y_train, y_test

def download_s3_data(filename):
    """Downloads npz file from s3 to the 'trees_temp' directory.

    ARGUMENTS:
    - filename (string): name of file to retrieve from S3 bucket
    """
    conn, bucket = create_connection('treedata-ks')
    key = bucket.get_key('test_train_data/' + filename)
    key.get_contents_to_filename('trees_temp/' + filename)

def get_data(num_channels, training_size):
    """Checks if data is local, downloads if not.

    ARGUMENTS:
    - num_channels (int): specifies whether to find grayscale (1) or color (3)
    - training_size (int): limits amount of data to train

    RETURNS:
    (array, array): (features, labels)
    """
    if num_channels == 1:
        filename = 'test_train_data.npz'
    if num_channels == 3:
        filename = 'test_train_data_color.npz'
    if not os.path.exists('trees_temp/' + filename):
        print('Downloading data from S3.')
        download_s3_data(filename)
    X_train, X_test, y_train, y_test = restore_matrices('trees_temp/' + filename)
    X_train = X_train[:training_size]
    y_train = y_train[:training_size]
    X_train = X_train.reshape(X_train.shape[0], 100, 100, num_channels)
    y_train = y_train.reshape(y_train.shape[0], 1)
    X_test = X_test.reshape(X_test.shape[0], 100, 100, num_channels)
    y_test = y_test.reshape(y_test.shape[0], 1)
    return X_train, X_test, y_train, y_test

def standardize(X):
    """Returns a standardized version of the input data.

    ARGUMENTS:
    X (array): feature data

    RETURNS:
    (array): centered feature data
    """
    centers = np.mean(X, axis=(0, 1, 2))
    X = X.astype('float32') - centers
    return X

def train_val_split(X, y):
    """Returns training and validation data.

    ARGUMENTS:
    X (array): feature data
    y (array): label data

    RETURNS:
    (array, array, array, array): (training features, validation features,
                                   training labels, validation labels)
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3)
    return X_train, X_val, y_train, y_val

class TreeIDModel(object):
    """
    """
    def __init__(self, X_train, #X_val,
                 y_train, #y_val,
                 num_epochs, batch_size=32, learning_rate=0.00001, alpha=0):
        """Initializes data and parameters for training a neural network.

        ARGUMENTS:
        - X_train (array): features of training data
        - X_val (array): features of validation data
        - y_train (array): labels for training data
        - y_val (array): labels for validation data
        - num_epochs (int): number of epochs for training model
        - batch_size (int): size of batch for training model
        - learning_rate (float): optimizer learning rate for training model
        - alpha (float): parameter for L2 regularization in VGG Conv2D layers
        """
        self.X_train = X_train
        #self.X_val = X_val
        self.y_train = y_train
        #self.y_val = y_val

        self.num_channels = self.X_train.shape[3]
        self.input_shape = (100, 100, self.num_channels)
        self.training_size = self.X_train.shape[0] #+ self.X_val.shape[0]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.vgg_flag = 0
        self.alpha = alpha
        self.metadata_string = self.get_metadata_string()

        self.model = None
        self.history = None

    def get_metadata_string(self):
        """Returns a metadata string used for naming saved files.
        """
        metadata_string = str(self.training_size) + 'images_' + \
                          str(self.num_epochs) + 'epochs_' + \
                          str(self.batch_size) + 'batch_' + \
                          str(self.learning_rate) + 'lr_' + \
                          str(self.alpha) + 'reg'
        if self.num_channels == 3: metadata_string += '_RGB'
        return metadata_string

    def validation_metrics(self, X, y, data_label):
        """Prints loss, accuracy, precision, and recall for the data.
        """
        score = self.model.evaluate(X, y, verbose=0)
        print(data_label + ' loss:' , score[0])

        y_pred = self.model.predict(X)

        accuracy = np.sum((y_pred >= 0.5) == (y == 1))/y.shape[0]
        print(data_label + ' accuracy: ', accuracy)

        TP = float(np.sum((y_pred  >= 0.5) & (y == 1)))
        FN = float(np.sum((y_pred < 0.5) & (y == 1)))
        FP = float(np.sum((y_pred >= 0.5) & (y != 1)))
        TN = float(np.sum((y_pred < 0.5) & (y != 1)))
        print(data_label + ' confusion matrix')
        print('     (TP, FN): ', (TP, FN))
        print('     (FP, TN): ', (FP, TN))

        if np.sum(y_pred >= 0.5) > 0:
            precision = TP/(TP + FP)
        else: precision = 'No predicted positives.'
        print(data_label + ' precision: ', precision)

        if np.sum(y == 1) > 0:
            recall = TP/(TP + FN)
        else: recall = 'No positive labels in data set.'
        print(data_label + ' recall: ', recall, '\n')

    def nn_model(self):
        """Runs a simple model, based on the MNIST example from Keras.
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=self.input_shape,
                              kernel_initializer='he_normal',
                              bias_initializer=Constant(0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, (3, 3), activation='relu',
                              kernel_initializer='he_normal',
                              bias_initializer=Constant(0.01)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu',
                             kernel_initializer='he_normal',
                             bias_initializer=Constant(0.01)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.25))
        self.model.add(Dense(1, activation='sigmoid',
                             kernel_initializer='glorot_normal'))

        self.model.compile(loss=binary_crossentropy,
                           optimizer=Adam(lr=self.learning_rate),
                           metrics=['accuracy'])

        self.history = self.model.fit(self.X_train, self.y_train,
                                 verbose=1,
                                 batch_size=self.batch_size,
                                 validation_data=(self.X_val, self.y_val),
                                 epochs=self.num_epochs)

        self.validation_metrics(self.X_train, self.y_train, 'training')
        self.validation_metrics(self.X_val, self.y_val, 'validation')

    def add_convolutional_layer(self, num_filters):
        """Adds a convolutional layer to the model.
        """
        self.model.add(Conv2D(num_filters, (3, 3), activation='relu',
                              kernel_initializer='he_normal',
                              bias_initializer=Constant(0.01),
                              kernel_regularizer=l2(self.alpha)))
        self.model.add(BatchNormalization())

    def vgg_model(self):
        """Runs modified version of VGG16 model. First two fully connected layer
        removed, and batch normalization added between layers.
        """
        self.model = Sequential()
        self.model.add(ZeroPadding2D((1, 1), input_shape=self.input_shape,
                                     data_format="channels_last"))
        self.add_convolutional_layer(64)
        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(64)
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(128)
        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(128)
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(256)
        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(256)
        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(256)
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(512)
        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(512)
        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(512)
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(512)
        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(512)
        self.model.add(ZeroPadding2D((1, 1)))
        self.add_convolutional_layer(512)
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # Add Fully Connected Layer
        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid',
                             kernel_initializer='glorot_normal'))

        self.model.compile(loss=binary_crossentropy,
                           optimizer=Adam(lr=self.learning_rate),
                           metrics=['accuracy'])

        self.history = self.model.fit(self.X_train, self.y_train,
                                      batch_size = self.batch_size,
                                      #validation_data=(self.X_val, self.y_val),
                                      epochs = self.num_epochs,
                                      verbose = 1)

        self.validation_metrics(self.X_train, self.y_train, 'training')
        #self.validation_metrics(self.X_val, self.y_val, 'validation')

        self.metadata_string += '_VGG'
        self.vgg_flag = 0

    def save_data(self):
        """Saves model and history to files.
        """
        model_filename = 'trees_temp/model_' + self.metadata_string + '.h5'
        self.model.save(model_filename)
        history_filename = 'trees_temp/hist_' + self.metadata_string + '.p'
        pickle.dump(self.history.history, open(history_filename, "wb" ))


def check_filepaths():
    """Creates temporary local directory if it doesn't exist.
    """
    if not os.path.exists('trees_temp'):
        os.makedirs('trees_temp')

def main(num_channels, training_size, num_epochs, batch_size, learning_rate,
         vgg_flag, alpha):
    """Sets up data and runs model. Saves model and epoch history locally.

    ARGUMENTS:
    - num_channels (int): specifies whether to find grayscale (1) or color (3)
    - training_size (int): limits amount of data to train
    - num_epochs (int): number of epochs for training model
    - batch_size (int): size of batch for training model
    - learning_rate (float): optimizer learning rate for training model
    - vgg_flag (int): specifies whether to run simple (0) or deep (1) model
    - alpha (float): parameter for L2 regularization in VGG Conv2D layers
    """
    check_filepaths()

    X_train, X_test, y_train, y_test = get_data(num_channels, training_size)
    standardize(X_train)
    X_train, X_val, y_train, y_val = train_val_split(X_train, y_train)

    treeID = TreeIDModel(X_train, X_val, y_train, y_val, num_epochs, batch_size,
                         learning_rate, alpha)
    if vgg_flag == 1:
        treeID.vgg_model()
    else:
        treeID.nn_model()
    treeID.save_data()

if __name__ == '__main__':
    num_channels = int(sys.argv[1]) # if grayscale, 1; if RGB, 3
    training_size = int(sys.argv[2])
    num_epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    learning_rate = float(sys.argv[5])
    vgg_flag = int(sys.argv[6])
    alpha = float(sys.argv[7])
    main(num_channels, training_size, num_epochs, batch_size,
         learning_rate, vgg_flag, alpha)
