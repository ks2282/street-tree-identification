"""
Generates model using specified parameters, using all training data (no
validation holdout). This script should only be used after model parameters
have been tuned with different parameters.
"""
from model_training import get_data, TreeIDModel
import numpy as np
import pickle

def standardize_test_train(X_train, X_test):
    """Returns a standardized version of the input data.

    ARGUMENTS:
    X_train (array): feature data
    X_test (array): feature data for test set

    RETURNS:
    (array, array): centered feature data for (X_train, X_test)
    """
    centers = np.mean(X_train, axis=(0, 1, 2))
    X_train = X_train.astype('float32') - centers
    X_test = X_test.astype('float32') - centers
    return X_train, X_test

def predict_test(model, X_test, y_test):
    """

    """
    score = final_model.model.evaluate(X_test, y_test, verbose=0)
    print('test loss:' , score[0])

    y_pred = final_model.model.predict(X_test)

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
    print(data_label + 'test precision: ', precision)

    if np.sum(y == 1) > 0:
        recall = TP/(TP + FN)
    else: recall = 'No positive labels in data set.'
    print('test recall: ', recall, '\n')

    return y_pred

def save_data(model, X_test, y_test, y_pred):
    """Saves model and history to files.
    """
    model_filename = 'trees_temp/final_model_' + model.metadata_string + '.h5'
    final_model.model.save(model_filename)
    history_filename = 'trees_temp/final_hist_' + model.metadata_string + '.p'
    pickle.dump(final_model.history.history, open(history_filename, "wb" ))
    np.savez_compressed('trees_temp/test_X-y-pred', X_test, y_test, y_pred)

def main():
    X_train, X_test, y_train, y_test = get_data(3, 141750)
    X_train, X_test = standardize_test_train(X_train, X_test)
    final_model = TreeIDModel(X_train, y_train, num_epochs=15, batch_size=32,
                              learning_rate=0.00001, alpah=0)
    final_model.vgg_model()
    y_pred = predict_test(final_model, X_test, y_test)
    save_data(final_model, X_test, y_test, y_pred)

if __name__ == '__main__':
    main()
