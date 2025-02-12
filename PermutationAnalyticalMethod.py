"""
    Building on the work done in the Normalization.py file, 
    we can now apply the normalizations in the process of these analytical methods.
"""

import pandas as pd
import numpy as np
from itertools import permutations


class PermutationMethod:
    """
    Class to perform permutation-based method for calculating concordance matrices
    and net scores based on a normalized decision matrix and weights.
    """

    def __init__(self, normalized_matrix: pd.DataFrame, weights: list):
        """
        Initialize the PermutationMethod with the normalized matrix and weights.

        Args:
            normalized_matrix (pd.DataFrame): The normalized decision matrix.
            weights (list): A list of weights for each attribute.
        """
        self.normalized_matrix = normalized_matrix
        self.weights = weights
        self.concordance_matrices = {}
        self.permutation_scores = {}

    def calculate_concordance_matrix(self, permuted_alternatives: tuple) -> pd.DataFrame:
        """
        Calculate the concordance matrix for a given permutation of alternatives.

        Args:
            permuted_alternatives (list): A list of permuted alternatives.

        Returns:
            pd.DataFrame: The concordance matrix.
        """
        alternatives_num = self.normalized_matrix.shape[0]
        returning_concordance_matrix = pd.DataFrame(
            np.zeros((alternatives_num, alternatives_num)),
            index=permuted_alternatives,
            columns=permuted_alternatives,
        )
        for i in permuted_alternatives:
            for j in permuted_alternatives:
                if i != j:
                    self.update_concordance_matrix_cell(
                        matrix=returning_concordance_matrix,
                        i=i,
                        j=j,
                    )
        return returning_concordance_matrix

    def update_concordance_matrix_cell(self, matrix: pd.DataFrame, i, j):
        """
        Update a cell in the concordance matrix based on the comparison of two alternatives.

        Args:
            matrix (pd.DataFrame): The concordance matrix to be updated.
            i (str): The first alternative.
            j (str): The second alternative.
        """
        weight_sum = 0
        for attr in self.normalized_matrix.columns:
            if (
                self.normalized_matrix.loc[i, attr]
                >= self.normalized_matrix.loc[j, attr]
            ):
                weight_sum += self.weights[self.normalized_matrix.columns.get_loc(attr)]
        matrix.loc[i, j] = round(weight_sum, 2)

    def calculate_permutation_score(self, concordance_matrix: pd.DataFrame) -> float:
        """
        Calculate this permutation score, which is the difference between the sums of the upper and lower triangles of the concordance matrix.

        Args:
            concordance_matrix (pd.DataFrame): The concordance matrix of the corresponding permutation.

        Returns:
            float: The score.
        """
        upper_triangle_sum = (
            concordance_matrix.where(
                np.triu(np.ones(concordance_matrix.shape), k=1).astype(bool)
            )
            .sum()
            .sum()
        )
        lower_triangle_sum = (
            concordance_matrix.where(
                np.tril(np.ones(concordance_matrix.shape), k=-1).astype(bool)
            )
            .sum()
            .sum()
        )
        return round(upper_triangle_sum - lower_triangle_sum, 2)

    def run_calculations(self):
        """
        Run the permutation method calculations and save results to an output file.
        """
        options = self.normalized_matrix.index.tolist()
        # Permutations
        permutations_list = list(permutations(options))
        for permutation in permutations_list:
            if permutation not in self.permutation_scores:
                current_C_matrix = self.calculate_concordance_matrix(permutation)
                self.concordance_matrices[tuple(permutation)] = current_C_matrix
                self.permutation_scores[tuple(permutation)] = (
                    self.calculate_permutation_score(
                        concordance_matrix=current_C_matrix
                    )
                )
        self.write_output_file()

    def write_output_file(self):
        """
        Write the concordance matrices and scores for each permutation to an output file.
        
         The final line of the output contains the best permutation and its corresponding score.
        """
        with open("output.txt", "w") as file:
            for permutation, concordance_matrix in self.concordance_matrices.items():
                file.write(
                    f"""Permutation:\t{permutation:}\n{concordance_matrix.to_string()}\nScore:\t{self.permutation_scores.get(permutation)}\n{"*" * 50}\n"""
                )


if __name__ == "__main__":
    N = pd.read_csv("D_matrix.csv", skiprows=1)
    N.index = [f"A{i+1}" for i in range(len(N))]
    W = pd.read_csv("D_matrix.csv", nrows=1, header=None).iloc[0].tolist()
    N = N.drop("OP", axis=1)
    print("Normalized Decision Matrix:")
    print(N)
    print("Weights:")
    print(W)
    pm = PermutationMethod(normalized_matrix=N, weights=W)
    pm.run_calculations()
