import numpy as np
import sklearn.metrics as skm
from mysklearn.myevaluation import accuracy_score,  precision_score, recall_score, f1_score

a_y_pred = ['cat', 'dog', 'horse', 'cat', 'cat', 'dog', 'horse', 'horse']
a_y_true = ['cat', 'dog', 'cat', 'cat', 'dog', 'dog', 'horse', 'horse']
a_labels = ['cat', 'dog', 'horse']

b_y_pred = ['true', 'false', 'true', 'true', 'false', 'true', 'false', 'false']
b_y_true = ['false', 'true', 'true', 'false', 'false', 'true', 'true', 'false']
b_labels = ['true', 'false']

c_y_pred = [0, 1, 2, 3, 2, 1, 2, 3, 0, 0, 1, 2, 3, 1, 2, 2, 3, 1, 0, 1, 2, 0]
c_y_true = [1, 2, 3, 0, 2, 1, 3, 2, 1, 0, 0, 1, 2, 1, 3, 0, 1, 3, 2, 1, 2, 0]
c_labels = [0, 1, 2, 3]


def test_accuracy_score():
    print(accuracy_score(a_y_true, a_y_pred, a_labels))
    print(accuracy_score(b_y_true, b_y_pred, b_labels))
    print(accuracy_score(c_y_true, c_y_pred, c_labels))

    assert np.isclose(
        accuracy_score(a_y_true, a_y_pred, a_labels),
        0.833, atol=0.001)

    assert np.isclose(
        accuracy_score(b_y_true, b_y_pred, b_labels),
        0.5, atol=0.001)

    assert np.isclose(
        accuracy_score(c_y_true, c_y_pred, c_labels),
        0.659, atol=0.001)


def test_precision_score():
    assert np.isclose(
        precision_score(a_y_true, a_y_pred, a_labels),
        skm.precision_score(a_y_true, a_y_pred, average="macro"), atol=0.001)

    assert np.isclose(
        precision_score(b_y_true, b_y_pred, b_labels),
        skm.precision_score(b_y_true, b_y_pred, average="macro"), atol=0.001)

    assert np.isclose(
        precision_score(c_y_true, c_y_pred, c_labels),
        skm.precision_score(c_y_true, c_y_pred, average="macro"), atol=0.001)


def test_recall_score():
    assert np.isclose(
        recall_score(a_y_true, a_y_pred, a_labels),
        skm.recall_score(a_y_true, a_y_pred, average="macro"), atol=0.001)

    assert np.isclose(
        recall_score(b_y_true, b_y_pred, b_labels),
        skm.recall_score(b_y_true, b_y_pred, average="macro"), atol=0.001)

    assert np.isclose(
        recall_score(c_y_true, c_y_pred, c_labels),
        skm.recall_score(c_y_true, c_y_pred, average="macro"), atol=0.001)


def test_f1_score():
    assert np.isclose(
        f1_score(a_y_true, a_y_pred, a_labels),
        skm.f1_score(a_y_true, a_y_pred, average="macro"), atol=0.001)

    assert np.isclose(
        f1_score(b_y_true, b_y_pred, b_labels),
        skm.f1_score(b_y_true, b_y_pred, average="macro"), atol=0.001)

    assert np.isclose(
        f1_score(c_y_true, c_y_pred, c_labels),
        skm.f1_score(c_y_true, c_y_pred, average="macro"), atol=0.001)
