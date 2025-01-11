"""
    Based on what we have done in the Normalization.py file ,
    now we can have that normalizations and use in the process of these analytical method
"""

import pandas as pd
import numpy as np
from math import pow, factorial
from itertools import permutations


# TODO There is sth wrong with the concordance matrix creation, run then solve it

class PermutationMethod:
    def __init__(self, normalized_matrix: pd.DataFrame, weights: list):
        # The D Matrix should be created here
        self.normalized_matrix = normalized_matrix
        self.weights = weights
        self.concoradance_matrices = []
        # self.__concordance_matrices = self.calculate_concordance_matrix()

    def calculate_concordance_matrix(self, permutated_options) -> pd.DataFrame:
        # print("*" * 80, f"\nConcordance Matrix is being Created:")
        alternatives_num = self.normalized_matrix.shape[0]
        returning_concordance_matrix = pd.DataFrame(
            np.zeros((alternatives_num, alternatives_num)),
            index=permutated_options,
            columns=permutated_options,
        )
        flag = False
        if permutated_options == (0, 2, 3, 1):
            flag = True
        else:
            flag = False
        # print(returning_concordance_matrix)
        for i in range(len(permutated_options)):
            for j in range(len(permutated_options)):
                if i != j:
                    weight_sum = 0
                    for attr in self.normalized_matrix.columns:
                        if (
                            self.normalized_matrix.iloc[permutated_options[i]][attr]
                            >= self.normalized_matrix.iloc[permutated_options[j]][attr]
                        ):
                            if flag:
                                print(f"Weight Sum {weight_sum}")
                                print(
                                    f"The row {permutated_options[i]} is better than row {permutated_options[j]} in {attr}"
                                )
                            weight_sum += self.weights[
                                self.normalized_matrix.columns.get_loc(attr)
                            ]
                    returning_concordance_matrix.iloc[i, j] = weight_sum
                    if flag:
                        print(f"adding {weight_sum} to index {i,permutated_options[j]}")
                        print(
                            f"So the Updated C Matrix is \n{returning_concordance_matrix}"
                        )
        return returning_concordance_matrix

    def calculate_net_score(self, permutation):
        pass

    def run_calculations(self):
        options = self.normalized_matrix.index.tolist()
        permutation_scores = {}
        # Permutation
        permutations_list = list(permutations(options))
        for permutation in permutations_list:
            # print(len(permutation_scores.keys()))
            if permutation not in permutation_scores:
                if permutation == (0, 2, 3, 1):
                    print(permutation)
                    self.concoradance_matrices.append(
                        self.calculate_concordance_matrix(permutation)
                    )
                    print(self.concoradance_matrices[-1])
                    continue

                # print(f"Permuation ({permutation}) is being scored")
                # if(permuation ==
                self.concoradance_matrices.append(
                    self.calculate_concordance_matrix(permutation)
                )
                # print(self.concoradance_matrices[-1])
                permutation_scores[tuple(permutation)] = 0
                # self.calculate_concordance_matrix(
                #     permutated_options=current_permutation
                # )
                # self.calculate_concordance_matrix()
                # score
                # adding the score to the dict with this key
        with open("output.txt", "w") as file:
            for m in self.concoradance_matrices:
                file.write(m.to_string())
                file.write("\n\n")
                file.write("*" * 10)
                file.write("\n\n")
        # for each permutation there should be a C Matrix and a score
        # There must a dict of scores
        # find the highest score


if __name__ == "__main__":
    N = pd.read_csv("D_matrix.csv", skiprows=1)
    W = pd.read_csv("D_matrix.csv", nrows=1, header=None).iloc[0].tolist()
    N = N.drop("OP", axis=1)
    print("Normalized Decision Matrix:")
    print(N)
    print("Weights:")
    print(W)
    pm = PermutationMethod(normalized_matrix=N, weights=W)
    pm.run_calculations()
