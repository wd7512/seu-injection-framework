import numpy as np
from sklearn.metrics import accuracy_score

def classification_accuracy(model, X_tensor, y_true, device = None):

    if device:
        X_tensor = X_tensor.to(device)
        model = model.to(device)

    model.eval()

    y_pred = model(X_tensor)
    return multiclass_classification_accuracy(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

def multiclass_classification_accuracy(y_true, model_output):
    if model_output.ndim == 1 or model_output.shape[1] == 1:
        y_low = np.min(y_true)
        y_high = np.max(y_true)
        midpoint = (y_high + y_low) / 2

        y_pred = np.zeros_like(y_true) + y_low
        y_pred[model_output >= midpoint] = y_high
    else:
        y_pred = np.argmax(model_output, axis=1)
    return accuracy_score(y_true=y_true, y_pred=y_pred)
