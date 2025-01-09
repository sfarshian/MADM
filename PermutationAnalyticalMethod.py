"""
    Based on what we have done in the Normalization.py file ,
    now we can have that normalizations and use in the process of these analytical method
"""

import pandas as pd


class PermutationMethod:
    __decision_matrix: pd.DataFrame = None
    __normalized_matrix: pd.DataFrame = None
    __concordance_matrices: list = None

    @staticmethod
    def get_decision_matrix() -> pd.DataFrame:
        """Public method to access the static property"""
        return PermutationMethod.__decision_matrix

    @staticmethod
    def get_normalized_matrix() -> pd.DataFrame:
        """Public method to access the static property"""
        return PermutationMethod.__normalized_matrix

    @staticmethod
    def get_concordance_matrices() -> list:
        """Public method to access the static property"""
        return PermutationMethod.__concordance_matrices

    def __init__(self):
        # The D Matrix should be created here
        pass
