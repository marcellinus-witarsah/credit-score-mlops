"""
A module for weight of evidence calculation.
"""

from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WOETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_columns: list, categorical_columns: list, bins: int):
        """
        Initialize `WOETransformer` class.

        Args:
            numerical_columns (list): list of numerical column names.
            categorical_columns (list): categorical column names from the dataset.
            bins (int): number of bins.
        """
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.bins = bins
        self.woe_dict = {}
        self.woe_df = pd.DataFrame()
        self.iv_df = pd.DataFrame()

    def __generate_bins(
        self, df: pd.DataFrame, numerical_columns: list, bins: int
    ) -> pd.DataFrame:
        """
        Create bins for a numerical column, dividing it into a specified number of equal-sized bins.

        Args:
            df (pd.DataFrame): Pandas DataFrame containing the data.
            numerical_columns (str): list of numerical column names.
            bins (int): number of bins.
        Returns:
            pd.DataFrame: Pandas DataFrame with `numerical_column` values are changed to bin.
        """
        for numerical_column in numerical_columns:
            df[numerical_column] = pd.qcut(df[numerical_column], q=bins, duplicates="drop")
        return df

    def __fill_missing_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing categorical columns inside Pandas DataFrame with `Missing`. All data must already been binned.

        Args:
            df (pd.DataFrame): Pandas DataFrame containing the data.
        Returns:
            pd.DataFrame: Pandas DataFrame with `numerical_column` values are changed to bin.
        """
        for column in df.columns:
            if df[column].isna().sum() > 0 and df[column].dtype in [
                "object",
                "category",
            ]:
                # Add category 'Missing' to replace the missing values
                df[column] = df[column].cat.add_categories("Missing")
                # Replace missing values with category 'Missing'
                df[column] = df[column].fillna(value="Missing")
        return df

    def __get_woe(self, value: Union[int, float], column_name: str) -> float:
        """
        Tranform column values into weight of evidence value.

        Args:
            value (Union[int, float]): value of the column.
            column_name (str): column name to be mapped to WOE value.

        Returns:
            float: weight of evidence value.
        """
        woe_value = None
        if isinstance(value, int) or isinstance(value, float):
            for interval, woe in self.woe_dict[column_name].items():
                if isinstance(interval, pd.Interval) and value in interval:
                    woe_value = woe
        if isinstance(value, str):
            woe_value = self.woe_dict[column_name][value]
        return woe_value

    def __interpret_information_value(self, value: float) -> str:
        """
        Interpret the strength of the information value.

        Args:
            value (float): information value.

        Returns:
            str: interpretation of the information value.
        """
        if value < 0.02:
            return "Not Predictive"
        elif 0.02 <= value and value < 0.1:
            return "Weak Predictive"
        elif 0.1 <= value and value < 0.3:
            return "Medium Predictive"
        elif 0.3 <= value and value < 0.5:
            return "Strong Predictive"
        else:
            return "Very Strong Predictive"

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> Union[BaseEstimator, TransformerMixin]:
        """
        Perform binning, weight of evidence and information value calculation.

        Args:
            X (pd.DataFrame): Pandas DataFrame containing the data (predictor).
            y (pd.Series, optional): Pandas Series containing labels. Defaults to None.

        Returns:
            Union[BaseEstimator, TransformerMixin]: WOETransformer object
        """

        # 1. Perform binning on numerical column
        X_binned = self.__generate_bins(X.copy(deep=False), self.numerical_columns, self.bins)
        X_binned = self.__fill_missing_categorical(X_binned)

        # 2. Count the occurrences of the target inside each bin of the columns.
        crosstabs = {}
        for column in X_binned.columns:
            crosstabs[column] = pd.crosstab(X_binned[column], y, margins=True)

        # 3. Calculate weight of evidence for all columns
        for column, crosstab in crosstabs.items():
            # 3.1 Calculate WOE
            crosstab["proportion_not_default"] = crosstab[0] / crosstab[0]["All"]
            crosstab["proportion_default"] = crosstab[1] / crosstab[1]["All"]
            crosstab["WOE"] = np.log(
                crosstab["proportion_not_default"] / crosstab["proportion_default"]
            )
            temp_df = crosstab.reset_index().iloc[:-1, [0, -1]].copy()
            temp_df.columns = ["Value", "WOE"]

            self.woe_dict[column] = {
                row["Value"]: row["WOE"] for _, row in temp_df.iterrows()
            }  # for woe values mapping

            temp_df.loc[:, "Characteristic"] = column
            self.woe_df = pd.concat((self.woe_df, temp_df), axis=0)  # for pandas DataFrame

            # 3.2 Calculate information value
            IV = np.sum(
                (crosstab["proportion_not_default"] - crosstab["proportion_default"])
                * crosstab["WOE"]
            )
            self.iv_df = pd.concat(
                (
                    self.iv_df,
                    pd.DataFrame({"Characteristic": [column], "Information Value": [IV]}),
                ),
                axis=0,
            )
            self.iv_df["Interpretation"] = self.iv_df["Information Value"].apply(
                lambda x: self.__interpret_information_value(x)
            )
            self.iv_df = self.iv_df.sort_values(by="Information Value", ascending=False)
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """
        Mapping predictor values into its weight of evidence values.

        Args:
            X (pd.DataFrame): Pandas DataFrame containing the data (predictor).
            y (pd.Series, optional): Pandas Series containing labels. Defaults to None.

        Returns:
            pd.DataFrame: Transformed predictor variable in form of weight of evidence values.
        """
        X_transformed = X.copy(deep=False)
        for column in X.columns:
            X_transformed[column] = X_transformed[column].apply(
                lambda x: self.__get_woe(x, column)
            )
            X_transformed[column] = X_transformed[column].fillna(self.woe_dict.get("Missing", 0))
        return X_transformed
