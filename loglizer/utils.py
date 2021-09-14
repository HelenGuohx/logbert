"""
The utility functions of loglizer

Authors: 
    LogPAI Team

"""

from sklearn.metrics import precision_recall_fscore_support
import numpy as np


def metrics(y_pred, y_true):
    """ Calucate evaluation metrics for precision, recall, and f1.

    Arguments
    ---------
        y_pred: ndarry, the predicted result list
        y_true: ndarray, the ground truth label list

    Returns
    -------
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 measure value
    """
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum(y_true == 0) - TN
    FN = np.sum(y_true == 1) - TP
    precision = 100 * TP / (TP + FP + 1e-8)
    recall = 100 * TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)

    print(f"Confusion Matrix: TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    return precision, recall, f1


if __name__ == "__main__":
    print(metrics(np.array([1, 1, 1, 0, 0, 0]), np.array([1, 0, 1, 1, 0, 0])))

