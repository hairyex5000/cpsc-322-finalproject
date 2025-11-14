import numpy as np
from tabulate import tabulate
from mysklearn import myevaluation


def randomize_in_place(alist, parallel_list=None):
    """
    Randomly permute the elements of a list in place, optionally performing the
    same swaps on a parallel list to preserve element pairings.

    Args:
        alist(list): The list whose elements will be shuffled in place.
        parallel_list(list): If provided, must be the same length as `alist`. 
            Elements at the same indices will be swapped in 
            lockstep with `alist` so that paired data remain aligned.
    """
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        # rand int in [0, len(alist))
        rand_index = np.random.randint(0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]


def euclidean_distance(a: list, b: list) -> float:
    """
    Computes the Euclidean distance between two vectors.

    Args:
        a (list): The first vector.
        b (list): The second vector.

    Returns:
        float: The Euclidean distance between vectors a and b.

    Raises:
        ValueError: If the input vectors a and b are not of the same length.
    """
    if len(a) != len(b):
        raise ValueError("A and B must be the same length")

    return np.sqrt(
        np.sum([np.square(a[i] - b[i]) for i in range(len(a))])
    )


def discretize_mpg_doe(mpg: float) -> str:
    """
    Discretizes a given miles-per-gallon (mpg) value into categorical bins according to Department of Energy standards.

    Parameters:
        mpg (float): The miles-per-gallon value to discretize.

    Returns:
        str: A string representing the discrete bin label ("1" through "10").
    """
    if mpg <= 13:
        return "1"
    elif mpg <= 14:
        return "2"
    elif mpg <= 16:
        return "3"
    elif mpg <= 19:
        return "4"
    elif mpg <= 23:
        return "5"
    elif mpg <= 26:
        return "6"
    elif mpg <= 30:
        return "7"
    elif mpg <= 36:
        return "8"
    elif mpg <= 44:
        return "9"
    else:
        return "10"


def normalize_list(raw: list) -> list[float]:
    """
    Normalizes a list of numerical values to the range [0, 1].

    Args:
        raw (list): A list of numerical values to normalize.

    Returns:
        list[float]: A list of normalized float values in the range [0, 1].

    Notes:
        Assumes the list has at least 2 elements, all numerical, and are not identical.
    """
    r_min = min(raw)
    gap = max(raw) - r_min

    return [(v - r_min) / gap for v in raw]


def pretty_confusion_matrix(y_true, y_pred, labels):
    """
    Print a human-readable confusion matrix for given true and predicted labels.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        None

    Notes:
        The output is printed to the standard output.
    """
    matrix = myevaluation.confusion_matrix(y_true, y_pred, labels)

    print("Confusion Matrix")
    print(tabulate(matrix, headers=labels, showindex=labels))
