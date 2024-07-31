"""
A module for model evaluation.
"""

import pandas as pd
import typer
import dvc.api
from dvclive import Live
from pathlib import Path
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
    train_scores = model.evaluate(X_train, y_train, "Training")
    test_scores = model.evaluate(X_test, y_test, "Testing")

    # 5. Save results
    with Live(dir="reports", dvcyaml=None) as live:
        for metric_name, value in train_scores.items():
            live.log_metric(Path("train") / metric_name, value, plot=False)

        for metric_name, value in test_scores.items():
            live.log_metric(Path("test") / metric_name, value, plot=False)

        live.log_sklearn_plot(
            kind="calibration",
            labels=y_train,
            predictions=model.predict_proba(X_train)[:, 1],
            name="train_calibration",
            n_bins=10,
        )
        
        live.log_sklearn_plot(
            kind="calibration",
            labels=y_test,
            predictions=model.predict_proba(X_test)[:, 1],
            name="test_calibration",
            n_bins=10,
        )


if __name__ == "__main__":
    main()
