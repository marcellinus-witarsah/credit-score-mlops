"""
a module for data preprocessing.
"""

import time

import pandas as pd
import typer
from pathlib import Path
from sklearn.model_selection import train_test_split

from credit_score_mlops.utils import logger

from credit_score_mlops.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
import dvc.api


app = typer.Typer()


@app.command()
def main(
    raw_data_file: Path = RAW_DATA_DIR / "credit_risk_dataset.csv",
    train_file: Path = PROCESSED_DATA_DIR / "train.csv",
    test_file: Path = PROCESSED_DATA_DIR / "test.csv",
):
    start_time = time.perf_counter()
    logger.info("Split data")

    # 1. Load params:
    params = dvc.api.params_show()

    target = params["target"]
    test_size = params["data_preprocessing"]["test_size"]
    random_state = params["data_preprocessing"]["random_state"]

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
    logger.info("Split data finished in {:.2f} seconds.".format(elapsed_time))


if __name__ == "__main__":
    main()
