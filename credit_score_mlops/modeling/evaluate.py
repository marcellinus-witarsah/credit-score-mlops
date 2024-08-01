"""
A module for model evaluation.
"""

from pathlib import Path

import dvc.api
import pandas as pd
import typer

from credit_score_mlops.config import MODELS_DIR, PROCESSED_DATA_DIR
from credit_score_mlops.modeling import WOELogisticRegression
from credit_score_mlops.plots import plot_calibration_curve
from credit_score_mlops.utils import save_json

app = typer.Typer()


@app.command()
def main(
    model_file: Path = MODELS_DIR / "model.pkl",
    train_file: Path = PROCESSED_DATA_DIR / "train.csv",
    test_file: Path = PROCESSED_DATA_DIR / "test.csv",
):
    # 1. Load params:
    params = dvc.api.params_show()

    target = params["target"]
    train_metrics_file = params["evaluate"]["train_metrics_file"]
    test_metrics_file = params["evaluate"]["test_metrics_file"]
    train_calibration_curve_file = params["evaluate"]["train_calibration_curve_file"]
    test_calibration_curve_file = params["evaluate"]["test_calibration_curve_file"]

    # 2. Load data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    X_train, y_train = (
        train_df.drop(columns=[target]),
        train_df[target],
    )

    X_test, y_test = (
        test_df.drop(columns=[target]),
        test_df[target],
    )

    # 3. Initialize model
    model = WOELogisticRegression.from_file(model_file)

    # 4. Evaluate model performance
    train_metrics = model.evaluate(X_train, y_train, "Training")
    test_metrics = model.evaluate(X_test, y_test, "Testing")

    # 5. Save results
    save_json(data=train_metrics, path=train_metrics_file)
    plot_calibration_curve(
        y_true=y_train,
        y_pred_proba=model.predict_proba(X_train)[:, 1],
        model_name=model.__class__.__name__,
        path=train_calibration_curve_file,
    )
    save_json(data=test_metrics, path=test_metrics_file)
    plot_calibration_curve(
        y_true=y_test,
        y_pred_proba=model.predict_proba(X_test)[:, 1],
        model_name=model.__class__.__name__,
        path=test_calibration_curve_file,
    )


if __name__ == "__main__":
    main()
