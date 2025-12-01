import numpy as np
from mysklearn.myutils import euclidean_distance


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


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self._header = None
        self._attribute_domains = None
        self._classes = None
        self.tree = None

    def _select_attribute(self, instances, attributes):
        """Selects the best attribute to split on based on entropy.

        Args:
            instances(list of list of obj): The list of instances to evaluate.
            attributes(list of str): The list of attributes to consider for splitting.

        Returns:
            best_attribute(str): The attribute with the lowest entropy.
        """
        att_entropies = []

        for att in attributes:
            entropy = 0
            partitions = self._partition_instances(instances, att)

            # Calculate entropy for each partition
            for val in partitions.keys():
                subset = partitions[val]

                # Assume entropy of 0 if all same class
                if self._all_same_class(subset):
                    continue

                probs = {c: 0 for c in self._classes}

                for instance in subset:
                    c = instance[-1]
                    probs[c] = probs.get(c, 0) + 1

                for c in probs.keys():
                    probs[c] /= len(subset)

                entropy += -sum(p * np.log2(p) for p in probs.values()) * \
                    (len(subset) / len(instances))

            att_entropies.append(entropy)

        return attributes[np.argmin(att_entropies)]

    def _partition_instances(self, instances, attribute):
        """Partitions instances based on attribute values.

        Args:
            instances(list of list of obj): The list of instances to partition.
            attribute(str): The attribute to partition on.

        Returns:
            partitions(dict): A dictionary mapping attribute values to lists of instances.

        Notes:
            Taken from DecisionTreeFun.
        """
        att_index = self._header.index(attribute)
        att_domain = self._attribute_domains[attribute]
        partitions = {}
        for att_value in att_domain:  # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def _all_same_class(self, instances):
        """Checks if all instances belong to the same class.
        Args:
            instances(list of list of obj): The list of instances to check.

        Returns:
            bool: True if all instances have the same class label, False otherwise.

        Notes:
            Originally taken/modified from DecisionTreeFun.
        """
        # No instances means they are all the same class by default
        if len(instances) == 0:
            return True

        # get the class label of the first instance.
        first_class = instances[0][-1]
        for instance in instances:
            # if any label differs, return False immediately.
            if instance[-1] != first_class:
                return False

        # if the loop completes without finding differences, return True.
        return True

    def _majority_class(self, instances):
        """Finds the majority class label among the instances.
        Args:
            instances(list of list of obj): The list of instances to check.

        Returns:
            obj: The majority class label.

        Notes:
            If there is a tie, returns the first label alphabetically.
        """
        class_counts = {}
        for instance in instances:
            label = instance[-1]
            class_counts[label] = class_counts.get(label, 0) + 1

        max_c = -1
        max_label = None
        tie = False

        for label, count in class_counts.items():
            if count > max_c:
                max_c = count
                max_label = label
                tie = False
            elif count == max_c:
                tie = True

        if tie:
            # Default to first alphabetically
            s = sorted(class_counts.keys())[0]
            return s

        return max_label

    def _tdidt(self, instances, attributes):
        """Implements the TDIDT algorithm to build a decision tree.

        Args:
            instances(list of list of obj): The list of instances to build the tree from.
            attributes(list of str): The list of attributes to consider for splitting.

        Returns:
            tree(nested list): The decision tree represented as a nested list.

        Notes:
            Called recursively, arguments will be mutated by the function.
            Originally taken/modified from DecisionTreeFun.
        """
        split_att = self._select_attribute(instances, attributes)
        attributes.remove(split_att)

        tree = ["Attribute", split_att]
        partitions = self._partition_instances(instances, split_att)

        for att_value in sorted(partitions.keys()):
            subset = partitions[att_value]
            val_subtree = ["Value", att_value]

            if len(subset) > 0 and self._all_same_class(subset):
                # Case 1
                leaf = ["Leaf", subset[0][-1], len(subset), len(instances)]
                val_subtree.append(leaf)
            elif len(subset) > 0 and len(attributes) == 0:
                # Case 2
                leaf = ["Leaf", self._majority_class(
                    subset), len(subset), len(instances)]
                val_subtree.append(leaf)
            elif len(subset) == 0:
                # Case 3
                return "clash"
            else:
                # Recursive case
                subtree = self._tdidt(subset, attributes.copy())

                if subtree == "clash":
                    leaf = ["Leaf", self._majority_class(
                        subset), len(subset), len(instances)]
                    val_subtree.append(leaf)
                else:
                    val_subtree.append(subtree)

            tree.append(val_subtree)

        return tree

    def _predict_subtree(self, tree, instance):
        """Recursively predicts the class label for an instance using the given subtree.

        Args:
            subtree(nested list): The subtree to use for prediction.
            instance(list of obj): The instance to predict the class label for.

        Returns:
            obj: The predicted class label.

        Notes:
            Originally taken/modified from DecisionTreeFun.
        """
        data_type = tree[0]

        # Base case: if this is a leaf, just return its class label
        if data_type == "Leaf":
            label = tree[1]
            return label

        # Recursive case: if we are here, this is an Attribute node
        attribute_name = tree[1]
        attribute_index = self._header.index(attribute_name)
        instance_value = instance[attribute_index]

        # Look for the matching value node
        for values in tree[2:]:
            value = values[1]
            subtree = values[2]

            if instance_value == value:
                return self._predict_subtree(subtree, instance)

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # Start by building up headers and attribute domains programmatically
        self._header = ["att" + str(i) for i in range(len(X_train[0]))]
        self._attribute_domains = {a: set() for a in self._header}
        for row in X_train:
            for i in range(len(row)):
                self._attribute_domains[self._header[i]].add(row[i])

        self._classes = set()
        for label in y_train:
            self._classes.add(label)

        instances = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        self.tree = self._tdidt(instances, self._header.copy())

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self._predict_subtree(self.tree, i) for i in X_test]


class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        trees(list of MyDecisionTreeClassifier): The list of decision trees in the forest.
        N (int): The number of trees in the forest.
        M (int): The number of trees to use when making predictions.
        F (int): The number of features to randomly select during fitting.
        random_state(int): The seed for the random number generator.
    """

    def __init__(self, N, M, F, random_state=None):
        """Initializes the random forest classifier.

        Args:
            N (int): The number of trees in the forest.
            M (int): The number of trees to use when making predictions.
            F (int): The number of features to randomly select during fitting.
            random_state(int or None): The seed for the random number generator.
        """
        self.trees = []
        self.N = N
        self.M = M
        self.F = F
        self.random_state = random_state

    def fit(self, x_train, y_train):
        """Fits a random forest classifier.

        Args:
            x_train (list of list of obj): The training instances.
            y_train (list of obj): The target y values.
        """
        pass

    def predict(self, x_test):
        """Makes predictions for test instances in X_test.

        Args:
            x_test (list of list of obj): The list of testing samples.

        Returns:
            y_predicted (list of obj): The predicted target y values (parallel to X_test).
        """
        pass
