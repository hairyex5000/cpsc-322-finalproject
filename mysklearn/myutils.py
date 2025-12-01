import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from mysklearn import myevaluation
from mysklearn.mypytable import MyPyTable


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


def compute_random_subset(values, num_values):
    """
    Computes a random subset of specified size from the input list.

    Args:
        values (list): The list from which to select a random subset.
        num_values (int): The number of elements to include in the subset.

    Returns:
        list: A list containing a random subset of `num_values` elements from `values`.

    Notes:
        Taken from M6-A Lab Task 2.
    """
    values_copy = values[:]  # shallow copy
    np.random.shuffle(values_copy)  # in place shuffle
    return values_copy[:num_values]


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


def onehot_to_categorical(table: MyPyTable, classes: list[str]) -> MyPyTable:
    """
    Convert a table with one-hot encoded class columns into a table with a single categorical "Class" column.

    Args:
        table (MyPyTable): Source table.
        classes (list[str]): List of column names in `table.column_names` that represent one-hot
                    encoded classes.

    Returns:
        MyPyTable: A new table containing the preserved non-class columns and a single appended
        "Class" column containing the name of the active class for each row (when found).

    Notes:
        Assumes that all classes exist in the table and are at the right-most columns
    """
    out = MyPyTable()
    class_i = [table.column_names.index(v) for v in classes]

    # Determine new columns
    for c in table.column_names:
        if c not in classes:
            out.column_names.append(c)

    out.column_names.append("Class")

    for row in table.data:
        new = []

        # For each column
        for i in range(len(row)):
            if i not in class_i:
                new.append(row[i])
                continue

            if row[i] == 1:
                new.append(table.column_names[i])
                break

        out.data.append(new)

    return out


def group_by(table: MyPyTable, group_by_col: str) -> dict:
    """
    Group rows of a MyPyTable by the value in a specified column.

    Args:
        table (MyPyTable): Source table
        group_by_col (str): Name of column to group by. Must exist in table.

    Returns:
        dict: A dictionary mapping each distinct value found in the specified column to a list
        of rows that have that value.
    """
    out = {}
    col_index = table.column_names.index(group_by_col)

    for row in table.data:
        key = row[col_index]
        if key in out:
            out[key].append(row.copy())
        else:
            out[key] = [row.copy()]

    return out


def eda_group_data_by_class(table: MyPyTable, interest_col: str):
    interest_i = table.column_names.index(interest_col)
    grouped = group_by(table, "Class")

    for key in grouped.keys():
        for i in range(len(grouped[key])):
            grouped[key][i] = grouped[key][i][interest_i]

    return grouped
