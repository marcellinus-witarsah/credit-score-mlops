"""
a module for credit score scaling.
"""

import time
from typing import Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_score_mlops.utils import logger
from credit_score_mlops.woe_transformer import WOETransformer


class CreditScoreScaling:
    def __init__(
        self,
        pipeline: Pipeline,
        pdo: Union[int, float],
        odds: Union[int, float],
        scorecard_points: Union[int, float],
    ):
        """Initialize the CreditScoreScaling with model pipeline and scorecard parameters.

        Args:
            pipeline (Pipeline): The pipeline containing the trained logistic regression model and WOE transformer.
            pdo (Union[int, float]): Points to Double the Odds.
            odds (Union[int, float]): The odds (Bad/Good) at the scorecard points.
            scorecard_points (Union[int, float]): The scorecard points at the specified odds.
        """
        self.pipeline = pipeline
        self.intercept = self.pipeline[LogisticRegression.__name__].intercept_[0]
        self.n_features = len(self.pipeline[LogisticRegression.__name__].feature_names_in_)
        self.pdo = pdo
        self.odds = odds
        self.scorecard_points = scorecard_points
        self.factor = self.pdo / np.log(2)
        self.offset = self.scorecard_points - self.factor * np.log(self.odds)
        self.scorecard_map_dict = {}
        self.__generate_scorecard()

    def __generate_mapping(self, scorecard: pd.DataFrame) -> None:
        """
        Generate a mapping of scorecard features and their corresponding points.

        Args:
            scorecard (pd.DataFrame): The scorecard DataFrame containing characteristics, values, and points.
        """
        scorecard_features = scorecard["Characteristic"].unique()
        # 1. Create a dictionary mapping each feature to its corresponding points
        for feature in scorecard_features:
            self.scorecard_map_dict[feature] = {
                row["Value"]: row["Points"]
                for _, row in scorecard[scorecard["Characteristic"] == feature].iterrows()
            }

    def __get_credit_point(self, feature: str, value: Union[int, float, str]) -> float:
        """
        Get the credit point for a given feature and value.

        Args:
            feature (str): The feature name.
            value (Union[int, float, str]): The feature value.

        Returns:
            float: The credit point corresponding to the feature value.
        """
        credit_point = None
        # 1. Check if the value is a numeric type and find the corresponding credit point
        if isinstance(value, int) or isinstance(value, float):
            for interval, woe in self.scorecard_map_dict[feature].items():
                if isinstance(interval, pd.Interval) and value in interval:
                    credit_point = woe
            credit_point = (
                self.scorecard_map_dict[feature].get("Missing", None)
                if credit_point is None
                else credit_point
            )
        elif isinstance(value, str):
            # 2. If the value is a string, directly get the credit point from the mapping
            credit_point = self.scorecard_map_dict[feature][value]
        return credit_point

    def __generate_scorecard(self) -> pd.DataFrame:
        """
        Generate the scorecard from the model pipeline.

        Returns:
            pd.DataFrame: The generated scorecard with characteristics, Coefficients, WOE, and points.
        """
        start_time = time.perf_counter()
        # 1. Create a summary DataFrame of the model features and their coefficients
        model_summary = pd.DataFrame(
            {
                "Characteristic": self.pipeline[LogisticRegression.__name__].feature_names_in_,
                "Coefficient": self.pipeline[LogisticRegression.__name__].coef_.reshape(-1),
            }
        )

        # 2. Merge the model summary with the WOE DataFrame from the pipeline
        self.scorecard = pd.merge(
            left=model_summary,
            right=self.pipeline[WOETransformer.__name__].woe_df,
            how="left",
            on=["Characteristic"],
        )

        # 3. Calculate the points for each feature using the scorecard formula
        self.scorecard["Points"] = (self.offset / self.n_features) - self.factor * (
            (self.scorecard["WOE"] * self.scorecard["Coefficient"])
            + (self.intercept / self.n_features)
        )

        # 4. Generate the mapping for scorecard features and points
        self.__generate_mapping(self.scorecard)

        elapsed_time = time.perf_counter() - start_time
        logger.info("Generating scorecard finished in {:.2f} seconds.".format(elapsed_time))

    def __calculate_single_credit_score(self, data: dict) -> dict:
        """
        Calculate the credit score for a single instance.

        Args:
            data (dict): A dictionary of feature values for a single instance.

        Returns:
            dict: A dictionary with credit points for each feature and the total credit score.
        """
        # 1. Calculate credit points for each feature in the data
        details = {
            feature: self.__get_credit_point(feature, value) for feature, value in data.items()
        }

        # 2. Sum the credit points to get the total credit score
        details["credit_score"] = sum(details.values())
        return details

    def calculate_credit_score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the credit scores for a DataFrame of instances.

        Args:
            X (pd.DataFrame): The DataFrame containing feature values for multiple instances.

        Returns:
            pd.DataFrame: A DataFrame with credit points for each feature and the total credit score for each instance.
        """
        start_time = time.perf_counter()
        # 1. Apply the calculate_single_credit_score method to each instance in the DataFrame
        index = X.index
        results = X.apply(
            lambda instance: self.__calculate_single_credit_score(instance.to_dict()),
            axis=1,
        )
        elapsed_time = time.perf_counter() - start_time
        logger.info("Calculating credit score finished in {:.2f} seconds.".format(elapsed_time))
        return pd.DataFrame.from_dict(results.tolist()).set_index(index)
