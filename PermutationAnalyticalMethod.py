"""
    Based on what we have done in the Normalization.py file ,
    now we can have that normalizations and use in the process of these analytical method
"""

import pandas as pd
import numpy as np
from math import pow, factorial
from itertools import permutations


class PermutationMethod:

    @staticmethod
    def get_normalized_matrix() -> pd.DataFrame:
        """Public method to access the static property"""
        return PermutationMethod.__normalized_matrix

    @staticmethod
    def get_concordance_matrices() -> list:
        """Public method to access the static property"""
        return PermutationMethod.__concordance_matrices

    def __init__(self, normalized_matrix: pd.DataFrame, weights: list):
        # The D Matrix should be created here
        self.normalized_matrix = normalized_matrix
        self.weights = weights
        # self.__concordance_matrices = self.calculate_concordance_matrix()

    def calculate_concordance_matrix(self, permutated_options) -> pd.DataFrame:
        alternatives_num = self.normalized_matrix.shape[0]
        self.__concordance_matrices = pd.DataFrame(
            np.zeros((alternatives_num, alternatives_num)),
            index=self.normalized_matrix.index,
            columns=permutated_options,
        )
        print(self.__concordance_matrices)
        # for i in range(alternatives_num):
        #     for j in range(alternatives_num):
        #         if i != j:
        #             self.__concordance_matrices.iloc[i, j] = (
        #                 self.__normalized_matrix.iloc[i]
        #                 >= self.__normalized_matrix.iloc[j]
        #             ).sum()

    def calculate_net_score(self, permutation):
        pass

    def run_calculations(self):
        options = self.normalized_matrix.index.tolist()
        permutation_scores = {}
        # Permutation
        permutations_list = list(permutations(options))
        for permutation in permutations_list:
            print(len(permutation_scores.keys()))
            if tuple(permutation) not in permutation_scores:
                permutation_scores[tuple(permutation)] = 0
                print(permutation)
                # self.calculate_concordance_matrix(
                #     permutated_options=current_permutation
                # )
                # self.calculate_concordance_matrix()
                # score
                # adding the score to the dict with this key

        # for each permutation there should be a C Matrix and a score
        # There must a dict of scores
        # find the highest score


if __name__ == "__main__":
    data = {
        "Criterion1": [0.8, 0.6, 0.7],
        "Criterion2": [0.5, 0.9, 0.3],
        "Criterion3": [0.6, 0.4, 0.8],
    }
    normalized_matrix = pd.DataFrame(data, index=["A1", "A2", "A3"])
    print("Normalized Decision Matrix:")
    print(normalized_matrix)
    pm = PermutationMethod(normalized_matrix=normalized_matrix, weights=[0.2, 0.3, 0.5])
    pm.run_calculations()
