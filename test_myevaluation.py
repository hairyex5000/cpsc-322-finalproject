import numpy as np
import sklearn.metrics as skm
from mysklearn.myevaluation import accuracy_score,  precision_score, recall_score, f1_score


y_pred = ['cat', 'dog', 'horse', 'cat', 'cat', 'dog', 'horse', 'horse']
y_true = ['cat', 'dog', 'cat', 'cat', 'dog', 'dog', 'horse', 'horse']
labels = ['cat', 'dog', 'horse']


def test_accuracy_score():
    assert np.isclose(
        accuracy_score(y_true, y_pred, labels),
        skm.accuracy_score(y_true, y_pred), atol=0.001)


def test_precision_score():
    assert np.isclose(
        precision_score(y_true, y_pred, labels),
        skm.precision_score(y_true, y_pred, average="micro"), atol=0.001)


def test_recall_score():
    assert np.isclose(
        recall_score(y_true, y_pred, labels),
        skm.recall_score(y_true, y_pred, average="micro"), atol=0.001)


def test_f1_score():
    assert np.isclose(
        f1_score(y_true, y_pred, labels),
        skm.f1_score(y_true, y_pred, average="micro"), atol=0.001)
