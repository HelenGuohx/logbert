"""
The implementation of the one class SVM model for anomaly detection.

"""

import numpy as np
from sklearn import svm
from ..utils import metrics


class OneClassSVM(object):

    def __init__(self, kernel='rbf', degree=3, gamma='scale', nu=0.5, max_iter=-1):
        """ The one class SVM model for anomaly detection
        Arguments
        ---------
        See SVM API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html#sklearn.svm.OneClassSVM

        Attributes
        ----------
            classifier: object, the classifier for anomaly detection

        """
        self.classifier = svm.OneClassSVM(kernel=kernel, degree=degree, gamma=gamma, nu=nu, max_iter=max_iter)

    def fit(self, X, y):
        """
        Arguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """
        print('====== Model summary ======')
        self.classifier.fit(X, y)

    def predict(self, X):
        """ Predict anomalies with mined invariants

        Arguments
        ---------
            X: the input event count matrix

        Returns
        -------
            y_pred: ndarray, the predicted label vector of shape (num_instances,)
        """

        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        print('====== Evaluation summary ======')
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_pred, y_true)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
