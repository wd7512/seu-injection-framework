import numpy as np
from sklearn.metrics import accuracy_score

def classification_accuracy(model, X_tensor, y_true, device = None):

    if device:
        X_tensor.to(device)
        model.to(device)

    model.eval()

    y_pred = model(X_tensor)
    return multiclass_classification_accuracy(y_true, y_pred)

def multiclass_classification_accuracy(y_true, model_output):
    """
    y_true: array of shape (N,) with integer class labels [0..9]
    model_output: array of shape (N, 10) with predicted scores (logits or probabilities)
    """
    y_pred = np.argmax(model_output, axis=1)
    return accuracy_score(y_true=y_true, y_pred=y_pred)