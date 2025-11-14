import numpy as np
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myutils import euclidean_distance


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """

        # Create regressor if needed
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()

        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        raw = self.regressor.predict(X_test)

        # We need to classify the predictions using the discretizer

        return [self.discretizer(val) for val in raw]


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train: list[list], y_train: list):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test: list[list]) -> tuple[list[list[float]], list[list[int]]]:
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        results = ([], [])

        for test in X_test:
            dists = [euclidean_distance(test, other) for other in self.X_train]

            # Sort the indices based on their distance
            indices = sorted(list(range(len(self.X_train))),
                             key=lambda v: dists[v])

            results[0].append(sorted(dists))
            results[1].append(indices)

        return results

    def predict(self, X_test: list[list]) -> list:
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        results = []
        _, indicies = self.kneighbors(X_test)

        for i in range(len(X_test)):
            # Get top k neighbors classes
            top_classes = [self.y_train[indicies[i][j]]
                           for j in range(self.n_neighbors)]

            # Find the most common class in the top k
            vals, counts = np.unique(
                top_classes, return_counts=True, equal_nan=False)

            max_i = np.argmax(counts)

            results.append(vals[max_i])

        return results


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """

    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """

        # Compute the frequencies of the y_train
        classes, counts = np.unique(
            y_train, return_counts=True, equal_nan=False)

        # Determine the index with the highest count
        max_i = np.argmax(counts)

        self.most_common_label = classes[max_i]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        return [self.most_common_label for _ in range(len(X_test))]


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        classes(set of obj): All classes that occur in the training data.
        priors(dict): The prior probabilities computed for each
            label in the training set.
        conditionals(dict): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.classes = set()
        self.priors = {}
        self.conditionals = {}

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        # Tally the priors
        for v in y_train:
            self.priors[v] = self.priors.get(v, 0) + 1
            self.classes.add(v)

        # For each attribute (assume at least 1 instance)
        for i in range(len(X_train[0])):
            self.conditionals[i] = {}
            totals = {}

            # For each instance
            for j in range(len(X_train)):
                v = X_train[j][i]

                totals[v] = totals.get(v, 0) + 1

                # May need to create the new dict for the value
                if v not in self.conditionals[i].keys():
                    self.conditionals[i][v] = {}

                # Increment whichever class the result was
                self.conditionals[i][v][y_train[j]] = self.conditionals[i][v].get(
                    y_train[j], 0) + 1

            for k1 in self.conditionals[i]:
                for k2 in self.conditionals[i][k1]:
                    self.conditionals[i][k1][k2] /= self.priors[k2]

        # We saved this until the end to make computing the conditionals easier
        for key in self.priors.keys():
            self.priors[key] /= len(y_train)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        preds = []

        for inst in X_test:
            probs = {}

            for c in self.classes:
                p = self.priors[c]

                for i in self.conditionals.keys():
                    # A key error would indicate that the probability is 0
                    try:
                        p *= self.conditionals[i][inst[i]][c]
                    except KeyError:
                        p = 0

                probs[c] = p

            # Select the key with the highest probability
            preds.append(max(probs, key=probs.get))

        return preds
