"""
a module for data preprocessing.
"""

import time

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_score_mlops.config import DATA_PREPROCESSING
from credit_score_mlops.utils import logger


def main():
    start_time = time.perf_counter()
    logger.info("Split data")

    # 1. Load data:
    df = pd.read_csv(DATA_PREPROCESSING.raw_data_file)

    # 2. Separate between features and label
    X, y = (
        df.drop(columns=[DATA_PREPROCESSING.target]),
        df[DATA_PREPROCESSING.target],
    )

    # 3. Split Data:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=DATA_PREPROCESSING.test_size,
        random_state=DATA_PREPROCESSING.random_state,
    )

    # 4. Concat into a DataFrame:
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    # 5. Save data:
    train.to_csv(DATA_PREPROCESSING.train_file, index=False)
    test.to_csv(DATA_PREPROCESSING.test_file, index=False)

    elapsed_time = time.perf_counter() - start_time
    logger.info("Split data finished in {:.2f} seconds.".format(elapsed_time))


if __name__ == "__main__":
    main()
