"""
Predicts test data using final model and saves output. This script should only
be used after final model has been created and saved.
"""
from model_training import get_data
import keras, pickle, numpy as np

def standardize_test_data(X_train, X_test):
    """Returns a standardized version of the input data.

    ARGUMENTS:
    X_train (array): feature data
    X_test (array): feature data for test set

    RETURNS:
    (array): centered feature data for (X_test)
    """
    centers = np.mean(X_train, axis=(0, 1, 2))
    X_test = X_test.astype('float32') - centers
    return X_test

def predict_test(model, X_test, y_test):
    """

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

    if np.sum(y == 1) > 0:
        recall = TP/(TP + FN)
    else: recall = 'No positive labels in data set.'
    print('test recall: ', recall, '\n')

    return y_pred

def main():
    X_train, X_test, y_train, y_test = get_data(3, 141750)
    X_test = standardize_test_data(X_train, X_test)
    model = keras.models.load_model('trees_temp/final_model_141750images_13epochs_32batch_0.001lr_0.0reg_RGB_VGG_25dropout.h5')
    y_pred = predict_test(model, X_test, y_test)
    np.savez_compressed('trees_temp/test_X-y-pred', X_test, y_test, y_pred)

if __name__ == '__main__':
    main()
