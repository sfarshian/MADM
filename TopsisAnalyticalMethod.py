import pandas as pd
from enum import Enum
from math import sqrt


class CriteriaType(Enum):
    BENEFICIAL = "+"
    DETRIMENTAL = "-"


class TopsisMethod:
    """
    Class implementing the TOPSIS method for decision-making based on the weighted normalized matrix,
    with the ability to calculate ideal solutions and distances to those solutions, and finally sort
    the alternatives based on their ranks.

    Attributes:
        normalized_matrix (pd.DataFrame): The normalized decision matrix.
        weights (list): List of weights corresponding to each criterion.
        criteria_types (pd.DataFrame): A DataFrame indicating whether each criterion is beneficial or detrimental.
        distances (dict): A dictionary storing the Euclidean distances for each alternative.
        closeness (dict): A dictionary storing the closeness score for each alternative.
    """

    def __init__(
        self,
        normalized_matrix: pd.DataFrame,
        weights: list,
        criteria_types: pd.DataFrame,
    ):
        """
        Initializes the TOPSIS method with the normalized matrix, weights, and criteria types.

        Args:
            normalized_matrix (pd.DataFrame): The normalized decision matrix.
            weights (list): The weights for each criterion.
            criteria_types (pd.DataFrame): A DataFrame indicating whether each criterion is beneficial or detrimental.
        """
        self.normalized_matrix = normalized_matrix
        self.weights = weights
        self.criteria_types = criteria_types
        self.distances = {}
        self.closeness = {}

    def calculate_weighted_matrix(self) -> pd.DataFrame:
        """
        Multiplies the normalized matrix by the weights to get the weighted matrix.

        Returns:
            pd.DataFrame: The weighted matrix.
        """
        return self.normalized_matrix * self.weights

    def run(self):
        """
        Runs the TOPSIS method to calculate the positive and negative ideal solutions, distances, and
        closeness scores for each alternative. Outputs the results.
        """
        self.weighted_matrix = self.calculate_weighted_matrix()
        print(f"Weighted Matrix(V):\n{self.weighted_matrix}")

        self.find_ideal_solutions()
        print(f"\nPositive Ideal Solution:\n{self.a_plus}\n")
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
        print(f"Ranking Based on Closeness:\n{self.closeness}")

    def calculate_closeness(self, distance: tuple):
        """
        Calculates the closeness coefficient for an alternative based on its distance to the ideal solutions.

        Args:
            distance (tuple): A tuple containing the distances to the positive and negative ideal solutions.
            distance[0] = d+
            distance[1] = d-

        Returns:
            float: The closeness coefficient.
        """

        return round(distance[1] / (distance[0] + distance[1]), 2)

    def get_ideal_solutions(self) -> None:
        """
        Finds the positive and negative ideal solutions (a_plus and a_minus) for all criteria in the weighted matrix.
        """
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
        """
        Calculates the Euclidean distance between an alternative and the ideal positive and negative solutions.

        Args:
            alternative (str): The alternative (row) to calculate the distance for.

        Returns:
            tuple: A tuple containing the distance to the positive ideal solution and the distance to the negative ideal solution.
        """

        d_plus = d_minus = 0
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
        return sqrt(d_plus), sqrt(d_minus)


if __name__ == "__main__":
    # Simple Example
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

    weights_data = [0.2, 0.1, 0.3, 0.25, 0.15]

    criteria_types = pd.DataFrame(
        [["+", "+", "+", "+", "-"]],
        columns=n_matrix.columns,
    )
    tm = TopsisMethod(
        normalized_matrix=n_matrix, weights=weights_data, criteria_types=criteria_types
    )
    tm.run()
