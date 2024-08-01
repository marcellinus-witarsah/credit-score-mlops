"""
a module for data preprocessing.
"""

import time

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_score_mlops.config import GLOBAL_PARAMS, DATA_PREPROCESSING_PARAMS
from credit_score_mlops.utils import logger


def main():
    start_time = time.perf_counter()
    logger.info("Splitting data ...")

    # 1. Load params:
    random_state = GLOBAL_PARAMS.random_state
    target = GLOBAL_PARAMS.target
    train_file = GLOBAL_PARAMS.train_file
    test_file = GLOBAL_PARAMS.test_file
    raw_data_file = DATA_PREPROCESSING_PARAMS.raw_data_file
    test_size = DATA_PREPROCESSING_PARAMS.test_size

    # 2. Load data:
    df = pd.read_csv(raw_data_file)

    # 3. Separate between features and label
    X, y = (
        df.drop(columns=[target]),
        df[target],
    )

    # 4. Split Data:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=test_size,
        random_state=random_state,
    )

    # 5. Concat into a DataFrame:
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    # 6. Save data:
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)

    elapsed_time = time.perf_counter() - start_time
    logger.info("Splitting data finished in {:.2f} seconds.".format(elapsed_time))


if __name__ == "__main__":
    main()
