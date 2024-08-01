"""
A module for Weight of Evidence and Logistic Regression Model.
"""

import pickle
import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from credit_score_mlops.metrics import gini, ks, pr_auc, roc_auc
from credit_score_mlops.utils import logger
from credit_score_mlops.woe_transformer import WOETransformer


class WOELogisticRegression(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline: Pipeline):
        """
        Initialize the WOELogisticRegression with a given pipeline.

        Args:
            pipeline (Pipeline): The scikit-learn pipeline containing the WOE transformer and Logistic Regression.
        """
        self.pipeline = pipeline

    @classmethod
    def from_file(cls, file_path: Union[str, Path]):
        """
        Create an instance of WOELogisticRegression from a saved pipeline file.

        Args:
            file_path (Union[str, Path]): Path to the file containing the saved pipeline.

        Returns:
            WOELogisticRegression: An instance of WOELogisticRegression initialized with the loaded pipeline.
        """
        with open(file_path, "rb") as file:
            pipeline = pickle.load(file)
        logger.info("Load {} model from {} file".format(pipeline.__class__.__name__, file_path))
        return cls(pipeline)

    @classmethod
    def from_parameters(cls, woe_transformer_params: dict, logreg_params: dict):
        """
        Create an instance of WOELogisticRegression from parameters for WOE transformer and Logistic Regression.

        Args:
            woe_transformer_params (dict): Parameters for the WOE transformer.
            logreg_params (dict): Parameters for the Logistic Regression.

        Returns:
            WOELogisticRegression: An instance of WOELogisticRegression initialized with the created pipeline.
        """
        pipeline = Pipeline(
            [
                (
                    WOETransformer.__name__,
                    WOETransformer(**woe_transformer_params),
                ),
                (LogisticRegression.__name__, LogisticRegression(**logreg_params)),
            ]
        )
        logger.info("{} model created".format(pipeline.__class__.__name__))
        return cls(pipeline)

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Union[BaseEstimator, TransformerMixin]:
        """
        Fit the pipeline to the training data.

        Args:
            X (pd.DataFrame): Training feature data.
            y (pd.Series, optional): Training target data. Defaults to None.

        Returns:
            Union[BaseEstimator, TransformerMixin]: The fitted WOELogisticRegression instance.
        """
        start_time = time.perf_counter()
        self.pipeline.fit(X, y)
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "{} model training finished in {:.2f} seconds.".format(
                self.pipeline.__class__.__name__, elapsed_time
            )
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the fitted pipeline.

        Args:
            X (pd.DataFrame): Feature data for making predictions.

        Returns:
            pd.DataFrame: Predictions made by the pipeline.
        """
        start_time = time.perf_counter()
        y_pred = self.pipeline.predict(X)
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "{} model prediction finished in {:.2f} seconds.".format(
                self.pipeline.__class__.__name__, elapsed_time
            )
        )
        return y_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using the fitted pipeline.

        Args:
            X (pd.DataFrame): Feature data for making probability predictions.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        start_time = time.perf_counter()
        y_pred_proba = self.pipeline.predict_proba(X)
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "{} model prediction probability finished in {:.2f} seconds.".format(
                self.pipeline.__class__.__name__, elapsed_time
            )
        )
        return y_pred_proba

    def evaluate(self, y_true: pd.Series, y_pred_proba: pd.Series, type: str = "Training") -> dict:
        """
        Evaluate the pipeline on test data.

        Args:
            y_true (pd.Series): True labels for the test data.
            y_proba (pd.Series): Prediction probability from the test data.
            type (str, optional): Type of evaluation ("Training" or "Validation"). Defaults to "Training".

        Returns:
            dict: A dictionary containing key-value pair of ROC AUC, PR AUC, gini, and KS scores.
        """
        start_time = time.perf_counter()
        roc_auc_score = roc_auc(y_true, y_pred_proba)
        pr_auc_score = pr_auc(y_true, y_pred_proba)
        gini_score = gini(y_true, y_pred_proba)
        ks_score = ks(y_true, y_pred_proba)
        logger.info(
            "{} Performance >>> ROC AUC: {:.2f}, PR AUC: {:.2f}, GINI: {:.2f}, KS SCORE: {:.2f}".format(
                type,
                roc_auc_score,
                pr_auc_score,
                gini_score,
                ks_score,
            )
        )
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "{} model evaluation finished in {:.2f} seconds.".format(
                self.pipeline.__class__.__name__, elapsed_time
            )
        )
        return {
            "ROC AUC Score": roc_auc_score,
            "PR AUC Score": pr_auc_score,
            "Gini Score": gini_score,
            "Kolmogorov-Smirnov Score": ks_score,
        }

    def save(self, file: Path) -> None:
        """
        Save the pipeline to a file.

        Args:
            file (Path): Path to the file where the pipeline should be saved.
        """
        start_time = time.perf_counter()
        with open(file, "wb") as file:
            pickle.dump(self.pipeline, file)
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            "Save {} model finished in {:.2f} seconds.".format(
                self.pipeline.__class__.__name__, elapsed_time
            )
        )
