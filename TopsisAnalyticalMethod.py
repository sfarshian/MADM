import numpy as np
import pandas as pd
from enum import Enum
from math import sqrt


class CriteriaType(Enum):
    BENEFICIAL = "+"
    DETRIMENTAL = "-"


class TopsisMethod:

    def __init__(
        self,
        normalized_matrix: pd.DataFrame,
        weights: list,
        criteria_types: pd.DataFrame,
    ):
        self.normalized_matrix = normalized_matrix
        self.weights = weights
        self.criteria_types = criteria_types
        self.distances = {}
        self.closeness = {}

    def calculate_weighted_matrix(self) -> pd.DataFrame:
        return self.normalized_matrix * self.weights

    def run(self):
        self.weighted_matrix = self.calculate_weighted_matrix()
        self.find_ideal_solutions()
        print(f"Positive Ideal Solution:\n{self.a_plus}\n")
        print(f"Negative Ideal Solution:\n{self.a_minus}\n")
        for alternative in self.weighted_matrix.index:
            self.distances[alternative] = self.calculate_euclidean_distance(
                alternative=alternative
            )
            # print(f"{self.distances[-1]} with type of {type(self.distances[-1])}")
            self.closeness[alternative] = self.calculate_closeness(
                self.distances[alternative]
            )
        self.closeness = dict(
            sorted(self.closeness.items(), key=lambda item: item[1], reverse=True)
        )

    def calculate_closeness(self, distance: tuple):
        # distance[0] = d+
        # distance[1] = d-
        return round(distance[1] / (distance[0] + distance[1]), 2)

    def find_ideal_solutions(self) -> None:
        # Ideal Positive Solution
        a_plus = []
        # Ideal Negative Solution
        a_minus = []
        for col in self.weighted_matrix.columns:
            criteria_type = self.criteria_types.iloc[
                0, self.weighted_matrix.columns.get_loc(col)
            ]
            if criteria_type == CriteriaType.BENEFICIAL.value:
                a_plus.append(round(self.weighted_matrix[col].max().item(), 3))
                a_minus.append(round(self.weighted_matrix[col].min().item(), 3))
            else:
                a_plus.append(round(self.weighted_matrix[col].min().item(), 3))
                a_minus.append(round(self.weighted_matrix[col].max().item(), 3))
        self.a_plus = pd.DataFrame(a_plus).T
        self.a_plus.columns = self.weighted_matrix.columns
        self.a_minus = pd.DataFrame(a_minus).T
        self.a_minus.columns = self.weighted_matrix.columns

    def calculate_euclidean_distance(self, alternative: str) -> tuple[float, float]:
        d_plus = 0
        d_minus = 0
        for column in self.weighted_matrix.loc[alternative].index:
            d_plus += pow(
                (
                    self.weighted_matrix.loc[alternative, column]
                    - self.a_plus.loc[0, column]
                ),
                2,
            )
            d_minus += pow(
                (
                    self.weighted_matrix.loc[alternative, column]
                    - self.a_minus.loc[0, column]
                ),
                2,
            )
        d_minus = sqrt(d_minus)
        d_plus = sqrt(d_plus)
        return d_plus, d_minus


if __name__ == "__main__":
    n_matrix = pd.DataFrame(
        {
            "X1": [0.8, 1.0, 0.93, 1.03],
            "X2": [0.56, 1.0, 0.95, 1.05],
            "X3": [0.95, 0.86, 1.0, 0.92],
            "X4": [0.87, 0.94, 0.79, 1.0],
            "X5": [0.72, 0.85, 0.80, 0.90],
        },
        index=["A1", "A2", "A3", "A4"],
    )

    # Weights for the criteria
    weights_data = [0.2, 0.1, 0.3, 0.25, 0.15]

    # Criteria types: '+' for benefit criteria, '-' for cost criteria
    criteria_types = pd.DataFrame(
        [["+", "+", "+", "+", "-"]],
        columns=n_matrix.columns,
    )
    print(criteria_types)
    tm = TopsisMethod(
        normalized_matrix=n_matrix, weights=weights_data, criteria_types=criteria_types
    )
    tm.run()
