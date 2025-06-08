import numpy as np
from sklearn.metrics import accuracy_score

def binary_classifcation_accuracy(y_true, model_output):
    unique_bins = np.unique(y_true)
    assert np.unique(y_true).size == 2

    lo, hi = unique_bins[0], unique_bins[1]

    y_bool = model_output > ((lo + hi) / 2)
    y_pred = np.zeros_like(y_true)
    y_pred[y_bool] = hi
    y_pred[~y_bool] = lo

    return accuracy_score(y_true=y_true, y_pred=y_pred)

def multiclass_classification_accuracy(y_true, model_output):
    """
    y_true: array of shape (N,) with integer class labels [0..9]
    model_output: array of shape (N, 10) with predicted scores (logits or probabilities)
    """
    y_pred = np.argmax(model_output, axis=1)
    return accuracy_score(y_true=y_true, y_pred=y_pred)