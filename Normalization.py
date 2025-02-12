import math
import pandas as pd
from enum import Enum


class CriteriaType(Enum):
    BENEFICIAL = "+"
    DETRIMENTAL = "-"


class Normalizer:
    """
    Class for normalizing a decision matrix using various methods.

    Attributes:
        decision_matrix (pd.DataFrame): The decision matrix to be normalized.
        criteria_types (pd.DataFrame): The criteria types (either '+' or '-') for each column.
    """

    def __init__(
        self, decision_matrix: pd.DataFrame, criteria_types: pd.DataFrame = None
    ):
        """
        Initialize the Normalizer with the decision matrix and optional criteria types.

        Args:
            decision_matrix (pd.DataFrame): The decision matrix to normalize.
            criteria_types (pd.DataFrame, optional): The criteria types for max-min normalization.
        """
        self.decision_matrix = decision_matrix
        self.criteria_types = criteria_types

    def linear_normalization(self) -> pd.DataFrame:
        """
        Normalize the decision matrix using linear normalization (divide by column sum).

        Returns:
            pd.DataFrame: The normalized decision matrix.
        """
        return self._apply_normalization(normalization_type="linear")

    def euclidean_normalization(self) -> pd.DataFrame:
        """
        Normalize the decision matrix using Euclidean normalization (divide by Euclidean norm).

        Returns:
            pd.DataFrame: The normalized decision matrix.
        """
        return self._apply_normalization(normalization_type="euclidean")

    def max_min_normalization(self) -> pd.DataFrame:
        """
        Normalize the decision matrix using max-min normalization based on criteria types.

        Returns:
            pd.DataFrame: The normalized decision matrix.
        """
        return self._apply_normalization(normalization_type="max_min")

    def _apply_normalization(self, normalization_type: str) -> pd.DataFrame:
        """
        Helper function to apply different types of normalization to the decision matrix.

        Args:
            normalization_type (str): The type of normalization to apply.

        Returns:
            pd.DataFrame: The normalized decision matrix.
        """
        normalized_matrix = self.decision_matrix.copy().astype(float)

        for col_name in normalized_matrix.columns:
            if normalization_type == "linear":
                normalized_matrix[col_name] /= normalized_matrix[col_name].sum()
            elif normalization_type == "euclidean":
                col_sum = (normalized_matrix[col_name] ** 2).sum()
                normalized_matrix[col_name] /= math.sqrt(col_sum)
            elif normalization_type == "max_min":
                criteria_type = self.criteria_types.iloc[
                    0, normalized_matrix.columns.get_loc(col_name)
                ]
                if criteria_type == CriteriaType.DETRIMENTAL.value:
                    normalized_matrix[col_name] = (
                        normalized_matrix[col_name].min() / normalized_matrix[col_name]
                    )
                else:  # BENEFICIAL
                    normalized_matrix[col_name] /= normalized_matrix[col_name].max()

        return normalized_matrix.round(2)


if __name__ == "__main__":
    decision_matrix = pd.DataFrame(
        {
            "Cost": [100, 150, 200],
            "Quality": [80, 70, 90],
            "Time": [30, 40, 20],
        },
        index=["A1", "A2", "A3"],
    )
    criteria_types = pd.DataFrame([["-", "+", "+"]], columns=decision_matrix.columns)
    normalizer = Normalizer(decision_matrix, criteria_types)
    print("Normalization Methods:\n")
    print("Linear Normalization:")
    print(normalizer.linear_normalization())
    print("\nEuclidean Normalization:")
    print(normalizer.euclidean_normalization())
    print("\nMax-Min Normalization:")
    print(normalizer.max_min_normalization())
