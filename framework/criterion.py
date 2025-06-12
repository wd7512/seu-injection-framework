import numpy as np
from sklearn.metrics import accuracy_score
import torch


def classification_accuracy(model, X_tensor, y_true, device=None, batch_size=64):
    if device:
        model = model.to(device)
    model.eval()

    y_pred_list = []
    y_true_list = []

    if batch_size is None:
        batch_size = len(X_tensor)

    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            end = start + batch_size
            batch_X = X_tensor[start:end]
            if device:
                batch_X = batch_X.to(device)
            batch_pred = model(batch_X)
            y_pred_list.append(batch_pred.cpu().detach().numpy())
            y_true_list.append(y_true[start:end].cpu().detach().numpy())

    y_pred_all = np.concatenate(y_pred_list, axis=0)
    y_true_all = np.concatenate(y_true_list, axis=0)

    return multiclass_classification_accuracy(y_true_all, y_pred_all)

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