from mysklearn.myclassifiers import MyRandomForestClassifier

# interview dataset
header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
X_train_interview = [
    ["Senior", "Java", "no", "no"],  # False
    ["Senior", "Java", "no", "yes"],  # False
    ["Mid", "Python", "no", "no"],  # True
    ["Junior", "Python", "no", "no"],  # True
    ["Junior", "R", "yes", "no"],  # True
    ["Junior", "R", "yes", "yes"],  # False
    ["Mid", "R", "yes", "yes"],  # True
    ["Senior", "Python", "no", "no"],  # False
    ["Senior", "R", "yes", "no"],  # True
    ["Junior", "Python", "yes", "no"],  # True
    ["Senior", "Python", "yes", "yes"],  # True
    ["Mid", "Python", "no", "yes"],  # True
    ["Mid", "Java", "yes", "no"],  # True
    ["Junior", "Python", "no", "yes"]  # False
]
y_train_interview = ["False", "False", "True", "True", "True", "False",
                     "True", "False", "True", "True", "True", "True", "True", "False"]


def test_random_forest_fit():
    assert True is False  # TODO


def test_random_forest_predict():
    assert True is False  # TODO
