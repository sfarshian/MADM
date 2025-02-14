import pandas as pd
from Normalization import Normalizer


class electre:
    def __init__(
        self, decision_matrix: pd.DataFrame, criteria_type: pd.DataFrame, weights: list
    ):
        self.weights = weights
        # The Normalization should be Euclidean
        self.decision_matrix = decision_matrix
        normalizer = Normalizer(
            decision_matrix=decision_matrix, criteria_types=criteria_type
        )
        self.normalized_matrix = normalizer.euclidean_normalization()

        self.weighted = self.calculate_weighted_matrix()

        self.concordance_matrix = self.generate_concordance_matrix()
        self.discordance_matrix = self.generate_discordance_matrix()

        # Only zeros and ones
        self.F = self.calculate_F()
        self.G = self.calculate_G()

        # AKA E
        self.aggregate_matrix = self.calculate_aggregate_matrix()

    def calculate_weighted_matrix(self) -> pd.DataFrame:
        """
        Multiplies the normalized matrix by the weights to get the weighted matrix.

        Returns:
            pd.DataFrame: The weighted matrix.
        """
        return self.normalized_matrix * self.weights