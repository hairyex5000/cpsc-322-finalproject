from copy import deepcopy
from mysklearn import myutils, myclassifiers, mypytable

CLASSES = ["Pastry","Z_Scratch","K_Scratch","Stains","Dirtiness","Bumps","Other_Faults"]
CLASS_HEADER = "Class"

data = mypytable.MyPyTable().load_from_file("plate-data.csv")
data = myutils.onehot_to_categorical(data, CLASSES)
features = deepcopy(data.column_names)
features.pop(features.index(CLASS_HEADER))

binned = mypytable.MyPyTable(data.column_names, data.data)

for feature in features:
    if feature == "TypeOfSteel_A300" or feature == "TypeOfSteel_A400":
        continue

    col_values = data.get_column(feature)
    binned_col = myutils.equal_width_bin(col_values, 10)
    binned.replace_column(feature, binned_col)

normalized = mypytable.MyPyTable(data.column_names, data.data)

for feature in features:
    if feature == "TypeOfSteel_A300" or feature == "TypeOfSteel_A400":
        continue

    col_values = data.get_column(feature)
    normalized_col = myutils.normalize_list(col_values)
    normalized.replace_column(feature, normalized_col)

myutils.stratified_kfold_tester(myclassifiers.MyDecisionTreeClassifier(len(features)),
                                binned,
                                CLASS_HEADER,
                                CLASSES,
                                10)