from mysklearn.myclassifiers import MyRandomForestClassifier, MyDecisionTreeClassifier

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

# note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
# note: the attribute values are sorted alphabetically
tree_interview = \
    ["Attribute", "att0",
     ["Value", "Junior",
      ["Attribute", "att3",
                    ["Value", "no",
                        ["Leaf", "True", 3, 5]
                     ],
                    ["Value", "yes",
                        ["Leaf", "False", 2, 5]
                     ]
       ]
      ],
     ["Value", "Mid",
      ["Leaf", "True", 4, 14]
      ],
     ["Value", "Senior",
      ["Attribute", "att2",
                    ["Value", "no",
                        ["Leaf", "False", 3, 5]
                     ],
                    ["Value", "yes",
                        ["Leaf", "True", 2, 5]
                     ]
       ]
      ]
     ]


# LA7 (fake) iPhone purchases dataset
header_iphone = ["standing", "job_status", "credit_rating"]
X_train_iphone = [
    [1, 3, "fair"],  # no
    [1, 3, "excellent"],  # no
    [2, 3, "fair"],  # yes
    [2, 2, "fair"],  # yes
    [2, 1, "fair"],  # yes
    [2, 1, "excellent"],  # no
    [2, 1, "excellent"],  # yes
    [1, 2, "fair"],  # no
    [1, 1, "fair"],  # yes
    [2, 2, "fair"],  # yes
    [1, 2, "excellent"],  # yes
    [2, 2, "excellent"],  # yes
    [2, 3, "fair"],  # yes
    [2, 2, "excellent"],  # no
    [2, 3, "fair"]  # yes
]
y_train_iphone = ["no", "no", "yes", "yes", "yes", "no",
                  "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

tree_iphone = \
    ['Attribute', 'att0',
        ['Value', 1,
            ['Attribute', 'att1',
                ['Value', 1,
                    ['Leaf', 'yes', 1, 5]
                 ],
                ['Value', 2,
                    ['Attribute', 'att2',
                        ['Value', 'excellent',
                            ['Leaf', 'yes', 1, 2]
                         ],
                        ['Value', 'fair',
                            ['Leaf', 'no', 1, 2]
                         ]
                     ]
                 ],
                ['Value', 3,
                    ['Leaf', 'no', 2, 5]
                 ]
             ]
         ],
        ['Value', 2,
            ['Attribute', 'att2',
                ['Value', 'excellent',
                    ['Leaf', 'no', 4, 10]
                 ],
                ['Value', 'fair',
                    ['Leaf', 'yes', 6, 10]
                 ]
             ]
         ]
     ]


# Since the random forest is based on decision trees, it makes sense to also test the modified decision tree here

def test_decision_tree_classifier_fit():
    # 1 - Interview

    tree = MyDecisionTreeClassifier()
    tree.fit(X_train_interview, y_train_interview, [i for i in range(4)])
    assert tree_interview == tree.tree

    # 2 - LA7 Phones

    tree = MyDecisionTreeClassifier()
    tree.fit(X_train_iphone, y_train_iphone, [i for i in range(3)])
    assert tree_iphone == tree.tree


def test_decision_tree_classifier_predict():
    # 1 - Interview

    tree = MyDecisionTreeClassifier()
    tree.fit(X_train_interview, y_train_interview, [i for i in range(4)])
    predicts = tree.predict([["Junior", "Java", "yes", "no"],
                             ["Junior", "Java", "yes", "yes"]])

    assert predicts == ["True", "False"]

    # 2 - LA7 Phones

    tree = MyDecisionTreeClassifier()
    tree.fit(X_train_iphone, y_train_iphone, [i for i in range(3)])
    predicts = tree.predict([[2, 2, "fair"],
                             [1, 1, "excellent"]])

    assert predicts == ["yes", "yes"]


def test_random_forest_classifier_fit():
    assert True is False  # TODO


def test_random_forest_classifier_predict():
    assert True is False  # TODO
