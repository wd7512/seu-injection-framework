import numpy as np
from sklearn.metrics import accuracy_score
import torch


def classification_accuracy_loader(model, data_loader, device=None):
    model.eval()
    if device:
        model = model.to(device)

    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            if device:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
            batch_pred = model(batch_X)
            y_pred_list.append(batch_pred)
            y_true_list.append(batch_y)

    y_pred_all = torch.cat(y_pred_list).cpu().numpy()
    y_true_all = torch.cat(y_true_list).cpu().numpy()
    return multiclass_classification_accuracy(y_true_all, y_pred_all)


def classification_accuracy(model, X_tensor, y_true, device=None, batch_size=64):
    if device:
        model = model.to(device)
        X_tensor = X_tensor.to(device)
        y_true = y_true.to(device)

    model.eval()
    y_pred_list = []
    y_true_list = []

    if batch_size is None:
        batch_size = len(X_tensor)

    with torch.no_grad():
        for start in range(0, len(X_tensor), batch_size):
            end = start + batch_size
            batch_X = X_tensor[start:end]
            batch_y = y_true[start:end]
            batch_pred = model(batch_X)
            y_pred_list.append(batch_pred)
            y_true_list.append(batch_y)

    y_pred_all = torch.cat(y_pred_list).cpu().numpy()
    y_true_all = torch.cat(y_true_list).cpu().numpy()

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
