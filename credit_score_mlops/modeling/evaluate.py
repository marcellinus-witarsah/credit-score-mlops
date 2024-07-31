"""
A module for model evaluation.
"""

import pandas as pd
import typer
import dvc.api
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
    train_scores_file = params["evaluate"]["train_scores_file"]
    test_scores_file = params["evaluate"]["test_scores_file"]
    # train_calibration_curve_file = params["evaluate"]["train_calibration_curve_file"]
    # test_calibration_curve_file = params["evaluate"]["test_calibration_curve_file"]

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
    # y_score_train = model.predict_proba(X_train)[:, 1]
    # y_score_test = model.predict_proba(X_test)[:, 1]
    train_scores = model.evaluate(X_train, y_train, "Training")
    test_scores = model.evaluate(X_test, y_test, "Testing")

    # 5. Save results
    save_json(data=train_scores, path=train_scores_file)
    # plot_calibration_curve(
    #     y_true=y_train,
    #     y_pred_proba=model.predict_proba(X_train)[:, 1],
    #     model_name=model.__class__.__name__,
    #     path=train_calibration_curve_file,
    # )
    save_json(data=test_scores, path=test_scores_file)
    # plot_calibration_curve(
    #     y_true=y_test,
    #     y_pred_proba=model.predict_proba(X_test)[:, 1],
    #     model_name=model.__class__.__name__,
    #     path=test_calibration_curve_file,
    # )


if __name__ == "__main__":
    main()
