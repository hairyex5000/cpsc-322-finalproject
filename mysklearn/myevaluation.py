import random
import numpy as np  # use numpy's random number generation
from mysklearn import myutils, mypytable

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state != None:
        np.random.seed(random_state)

    if shuffle:
        myutils.randomize_in_place(X, y)

    if isinstance(test_size, int):
        split_i = len(X) - test_size
    else:
        split_i = len(X) - int(len(X) * test_size) - 1

    return X[:split_i], X[split_i:], y[:split_i], y[split_i:]


def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state != None:
        np.random.seed(random_state)

    # Makes shuffling easier
    indices = [v for v in range(len(X))]

    if shuffle:
        myutils.randomize_in_place(indices)

    n_improp_folds = len(X) % n_splits
    folds = []

    for i in range(n_splits):
        last_end = 0
        train = []
        test = []

        for k in range(n_splits):
            size = len(X) // n_splits

            if k < n_improp_folds:
                size += 1

            if i == k:
                test.extend(indices[last_end:last_end + size])
            else:
                train.extend(indices[last_end:last_end + size])

            last_end += size

        folds.append((train, test))

    return folds


def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    x_len = len(X)
    to_map = {k: k for k in range(x_len)}
    if shuffle:
        random.seed(random_state)
        tmp_idx = []
        while len(tmp_idx) < x_len:
            tmp_rand = random.randint(0, x_len-1)
            if tmp_rand not in tmp_idx:
                tmp_idx.append(tmp_rand)
        to_map = {k: tmp_idx[k] for k in range(x_len)}

    tmp_dist = {}
    for index in range(len(y)):  # pylint: disable=consider-using-enumerate
        if y[index] in tmp_dist:
            tmp_dist[y[index]].append(to_map[index])
        else:
            tmp_dist[y[index]] = [to_map[index]]
    tmp_fold = [[] for x in range(n_splits)]
    for class_name in list(sorted(tmp_dist.keys(), key=lambda x: tmp_dist[x])):
        i = 0
        while len(tmp_dist[class_name]) > 0:
            tmp_fold[i % n_splits].append(tmp_dist[class_name].pop(-1))
            i += 1
    to_return = []
    for fold_index in range(n_splits):
        tmp_train = []
        for fold in (tmp_fold[0:fold_index] + tmp_fold[fold_index+1:]):
            for index in fold:
                tmp_train.append(index)
        to_return.append((tmp_train, tmp_fold[fold_index]))
    return to_return


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if random_state != None:
        np.random.seed(random_state)

    bag_indices = np.random.randint(0, len(X), len(
        X) if n_samples is None else n_samples)
    out_indices = []

    for i in range(len(X)):
        if i not in bag_indices:
            out_indices.append(i)

    if y is None:
        return [X[i] for i in bag_indices], [X[i] for i in out_indices], None, None
    else:
        return [X[i] for i in bag_indices], [X[i] for i in out_indices], [y[i] for i in bag_indices], [y[i] for i in out_indices]


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    true_i = [labels.index(v) for v in y_true]
    pred_i = [labels.index(v) for v in y_pred]

    matrix = np.zeros((len(labels), len(labels)))

    for i in range(len(true_i)):
        matrix[true_i[i]][pred_i[i]] += 1

    return [list(row) for row in matrix]


def accuracy_score(y_true, y_pred, labels):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels

    Returns:
        score(float): Predictive accuracy score as a ratio in [0.0, 1.0]

    Notes:
        Uses the macro approach.
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    matrix = np.array(confusion_matrix(y_true, y_pred, labels))

    # Macro approach: calculate accuracy per class and average
    accuracies = []
    for i in range(len(labels)):
        tp = matrix[i][i]
        tn = matrix.sum() - matrix[i].sum() - matrix[:, i].sum() + matrix[i][i]
        fp = matrix[:, i].sum() - matrix[i][i]
        fn = matrix[i].sum() - matrix[i][i]

        total = tp + tn + fp + fn
        if total > 0:
            accuracies.append((tp + tn) / total)
        else:
            accuracies.append(0)

    return np.mean(accuracies)


def precision_score(y_true, y_pred, labels):
    """
    Computes the precision score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels.

    Returns:
        precision(float): Precision score

    Notes:
        Uses the macro approach.
    """
    matrix = np.array(confusion_matrix(y_true, y_pred, labels))

    # Macro approach: calculate precision per class and average
    precisions = []
    for i in range(len(labels)):
        tp = matrix[i][i]
        fp = matrix[:, i].sum() - matrix[i][i]

        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0)

    return np.mean(precisions)


def recall_score(y_true, y_pred, labels):
    """
    Computes the recall score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels.

    Returns:
        recall(float): Recall score

    Notes:
        Uses the macro approach.
    """
    matrix = np.array(confusion_matrix(y_true, y_pred, labels))

    # Macro approach: calculate recall per class and average
    recalls = []
    for i in range(len(labels)):
        tp = matrix[i][i]
        fn = matrix[i].sum() - matrix[i][i]

        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0)

    return np.mean(recalls)


def f1_score(y_true, y_pred, labels):
    """
    Computes the F1 score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels.

    Returns:
        f1(float): F1 score

    Notes:
        Uses the macro approach.
    """
    matrix = np.array(confusion_matrix(y_true, y_pred, labels))

    # Macro approach: calculate F1 per class and average
    f1_scores = []
    for i in range(len(labels)):
        tp = matrix[i][i]
        fp = matrix[:, i].sum() - matrix[i][i]
        fn = matrix[i].sum() - matrix[i][i]

        # Calculate precision and recall for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate F1 for this class
        if precision + recall > 0:
            f1_scores.append(2 * (precision * recall) / (precision + recall))
        else:
            f1_scores.append(0)

    return np.mean(f1_scores)

def binary_error_rate(y_true, y_pred, labels=None, pos_label=None):
    """Calculates the error rate of classifications with binary labels.

    Args:
        y_true(list of obj): The actual classes that correspond to predictions made.
        y_pred(list of obj): Predictions made by the classifier.
        labels(list of obj): The two classes that are possible.
        pos_label(obj): The class that represents the positive label.

    Returns:
        score(float): The error rate of the predictions and actual classes provided.
    """
    if labels is None:
        labels = list(set(y_pred).union(set(y_true)))
    if pos_label is None:
        pos_label = sorted(list(set(labels)))[-1]
    fn = (sum(y_pred[x]!=pos_label and y_true[x]==pos_label for x in range(len(y_true))))
    fp = (sum(y_pred[x]==pos_label and y_true[x]!=pos_label for x in range(len(y_true))))
    if len(y_pred) != 0:
        return (fn+fp)/len(y_pred)
    return 0

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None, more_info=False):
    """Calculates the precision score of classifications with binary labels.

    Args:
        y_true(list of obj): The actual classes that correspond to predictions made.
        y_pred(list of obj): Predictions made by the classifier.
        labels(list of obj): The two classes that are possible.
        pos_label(obj): The class that represents the positive label.
        more_info(bool): Returns tuple with more info if desired

    Returns:
        score(float or dict): The precision score of the predictions and actual classes provided,
        or the precision score plus the true positives and false positives
    """
    if labels is None:
        labels = list(set(y_pred).union(set(y_true)))
    if pos_label is None:
        pos_label = sorted(list(set(labels)))[-1]
    tp = (sum(y_true[x]==pos_label and y_pred[x]==pos_label for x in range(len(y_true))))
    fp = (sum(y_pred[x]==pos_label and y_true[x]!=pos_label for x in range(len(y_true))))
    if tp+fp != 0:
        tmp_result = tp / (tp+fp)
    else:
        tmp_result = 0
    if more_info:
        return {
            'value': tmp_result,
            'tp': tp,
            'fp': fp
        }
    return tmp_result

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None, more_info=False):
    """Calculates the recall score of classifications with binary labels.

    Args:
        y_true(list of obj): The actual classes that correspond to predictions made.
        y_pred(list of obj): Predictions made by the classifier.
        labels(list of obj): The two classes that are possible.
        pos_label(obj): The class that represents the positive label.
        more_info(bool): Returns tuple with more info if desired

    Returns:
        score(float or dict): The recall score of the predictions and actual classes provided,
        or the recall score plus the true positives and false negatives
    """
    if labels is None:
        labels = list(set(y_pred).union(set(y_true)))
    if pos_label is None:
        pos_label = sorted(list(set(labels)))[-1]
    tp = (sum(y_true[x]==pos_label and y_pred[x]==pos_label for x in range(len(y_true))))
    fn = (sum(y_pred[x]!=pos_label and y_true[x]==pos_label for x in range(len(y_true))))

    if tp+fn != 0:
        tmp_result = tp / (tp+fn)
    else:
        tmp_result = 0
    if more_info:
        return {
            'value': tmp_result,
            'tp': tp,
            'fn': fn
        }
    return tmp_result

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Calculates the f1 score of classifications with binary labels.

    Args:
        y_true(list of obj): The actual classes that correspond to predictions made.
        y_pred(list of obj): Predictions made by the classifier.
        labels(list of obj): The two classes that are possible.
        pos_label(obj): The class that represents the positive label.

    Returns:
        score(float): The f1 score of the predictions and actual classes provided.
    """

    prec = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    if (prec+recall) != 0:
        return (2*prec*recall)/(prec+recall)
    return 0


def classification_report(y_true, y_pred, labels, output_dict=True):
    """Build a text report and a dictionary showing the main classification metrics.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        output_dict(bool): If True, return output as dict instead of a str

    Returns:
        report(str or dict): Text summary of the precision, recall, F1 score for each class.
            Dictionary returned if output_dict is True. Dictionary has the following structure:
                {'label 1': {'precision':0.5,
                            'recall':1.0,
                            'f1-score':0.67,
                            'support':1},
                'label 2': { ... },
                ...
                }
            The reported averages include macro average (averaging the unweighted mean per label) and
            weighted average (averaging the support-weighted mean per label).
            Micro average (averaging the total true positives, false negatives and false positives)
            multi-class with a subset of classes, because it corresponds to accuracy otherwise
            and would be the same for all metrics.

    Notes:
        Loosely based on sklearn's classification_report():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    """
    priors = {k: 0 for k in labels}
    for y_value in y_true:
        priors[y_value]+=1
    tmp_len = len(y_true)
    tp, fp, fn = 0, 0, 0
    prec, recall, f1 = 0, 0, 0
    sum_prec, sum_recall, sum_f1 = 0, 0, 0
    tmp_result = {}
    for class_name in priors:
        tmp_precision = binary_precision_score(y_true, y_pred, labels, class_name, True)
        tmp_recall = binary_recall_score(y_true, y_pred, labels, class_name, True)
        tmp_result[class_name] = {}
        tmp_result[class_name]['precision'] = tmp_precision['value']
        tmp_result[class_name]['recall'] = tmp_recall['value']
        tmp_result[class_name]['f1-score'] = binary_f1_score(y_true, y_pred, labels, class_name)
        tmp_result[class_name]['support'] = priors[class_name]
        prec+=tmp_result[class_name]['precision']
        recall+=tmp_result[class_name]['recall']
        f1+=tmp_result[class_name]['f1-score']
        tp+=tmp_precision['tp']
        fp+=tmp_precision['fp']
        fn+=tmp_recall['fn']
        sum_prec += tmp_result[class_name]['precision'] * (priors[class_name]/tmp_len)
        sum_recall += tmp_result[class_name]['recall'] * (priors[class_name]/tmp_len)
        sum_f1 += tmp_result[class_name]['f1-score'] * (priors[class_name]/tmp_len)

    tmp_result['micro avg'] = {}
    tmp_result['micro avg']['precision'] = tp / (tp+fp) if tp+fp != 0 else None
    tmp_result['micro avg']['recall'] = tp / (tp+fn) if tp+fn != 0 else None
    tmp_result['micro avg']['f1-score'] = (2*tmp_result['micro avg']['precision']*tmp_result['micro avg']['recall'])
    if tmp_result['micro avg']['precision']+tmp_result['micro avg']['recall'] != 0:
        tmp_result['micro avg']['f1-score'] /= tmp_result['micro avg']['precision']+tmp_result['micro avg']['recall']
    else:
        tmp_result['micro avg']['f1-score'] = 0
    tmp_result['micro avg']['support'] = len(y_true)

    tmp_result['macro avg'] = {}
    tmp_result['macro avg']['precision'] = prec/len(labels)
    tmp_result['macro avg']['recall'] = recall/len(labels)
    tmp_result['macro avg']['f1-score'] = f1/len(labels)
    tmp_result['macro avg']['support'] = len(y_true)

    tmp_result['weighted avg'] = {}
    tmp_result['weighted avg']['precision'] = sum_prec
    tmp_result['weighted avg']['recall'] = sum_recall
    tmp_result['weighted avg']['f1-score'] = sum_f1
    tmp_result['weighted avg']['support'] = len(y_true)

    if not output_dict:
        tmp_col_names = [""] + list(tmp_result['weighted avg'].keys())
        tmp_data = []
        for rows in tmp_result:
            tmp_data.append([rows] + list(map(lambda x: tmp_result[rows][x], tmp_col_names[1:])))
        return mypytable.MyPyTable(column_names=tmp_col_names, data=tmp_data).pretty_print()
    return tmp_result