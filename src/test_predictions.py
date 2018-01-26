"""
Predicts test data using final model and saves output. This script should only
be used after final model has been created and saved.
"""
from model_training import get_data, standardize
import keras, pickle, sys, numpy as np

def standardize_test_data(X_test, centers):
    """Returns a standardized version of the input data.

    ARGUMENTS:
    - X_test (array): feature data for test set
    - centers (array): channel means of training features

    RETURNS:
    (array): centered feature data for (X_test)
    """
    X_test = X_test.astype('float32') - centers
    return X_test

def predict_test(model, X_test, y_test):
    """Predicts class for test set, compares to test labels, and prints metrics.

    ARGUMENTS:
    - model (Keras model)
    - X_test (array): testing feature data
    - y_test (array): testing label data

    RETURNS:
    (array): predictions for test data
    """
    score = model.evaluate(X_test, y_test, verbose=0)
    print('test loss:' , score[0])

    y_pred = model.predict(X_test)

    accuracy = np.sum((y_pred >= 0.5) == (y_test == 1))/y_test.shape[0]
    print('test accuracy: ', accuracy)

    TP = float(np.sum((y_pred  >= 0.5) & (y_test == 1)))
    FN = float(np.sum((y_pred < 0.5) & (y_test == 1)))
    FP = float(np.sum((y_pred >= 0.5) & (y_test != 1)))
    TN = float(np.sum((y_pred < 0.5) & (y_test != 1)))
    print('test confusion matrix')
    print('     (TP, FN): ', (TP, FN))
    print('     (FP, TN): ', (FP, TN))

    if np.sum(y_pred >= 0.5) > 0:
        precision = TP/(TP + FP)
    else: precision = 'No predicted positives.'
    print('test precision: ', precision)

    if np.sum(y_test == 1) > 0:
        recall = TP/(TP + FN)
    else: recall = 'No positive labels in data set.'
    print('test recall: ', recall, '\n')

    return y_pred

def main(model_filepath, centers_filepath):
    """
    ARGUMENTS:
    - model_filepath (string)
    - centers_filepath (string)
    """
    X_train, X_test, y_train, y_test = get_data(3, 141750)
    centers = np.load(centers_filepath)
    X_test = standardize_test_data(X_test, centers)
    model = keras.models.load_model(model_filepath)
    y_pred = predict_test(model, X_test, y_test)
    np.savez_compressed('trees_temp/test_X-y-pred', X_test, y_test, y_pred)

if __name__ == '__main__':
    model_filepath = sys.argv[1]
    centers_filepath = sys.argv[2]
    main(model_filepath, centers_filepath)
