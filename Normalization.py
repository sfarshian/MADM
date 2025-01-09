import math
import pandas as pd

decision_matrix = pd.read_csv("D_matrix.csv", skiprows=1)
criteria_types = pd.read_csv("D_matrix.csv", nrows=1, header=None)
decision_matrix = decision_matrix.drop("Uni", axis=1)


def linear_normalization(dm: pd.DataFrame):
    n = dm.copy().astype(float)
    for col_name in n.columns:
        n[col_name] /= n[col_name].sum()
    return n.round(3)


def euclidean_normalization(dm: pd.DataFrame):
    n = dm.copy().astype(float)
    for col_name in n.columns:
        col_sum = (n[col_name] ** 2).sum()
        n[col_name] /= math.sqrt(col_sum)
    return n.round(3)


def max_min_normalization(dm: pd.DataFrame):
    n = dm.copy().astype(float)
    for col_name in n.columns:
        # criteria_types is defined before
        # accessing the criteria types df in order to find the type
        criteria_type = criteria_types.iloc[0, n.columns.get_loc(col_name)]
        if criteria_type == "-":
            n[col_name] = n[col_name].min() / n[col_name]
        # +
        else:
            n[col_name] /= n[col_name].max()
    return n.round(3)

