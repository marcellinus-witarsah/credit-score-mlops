"""
A module for model training.
"""

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import typer
from dotenv import find_dotenv, load_dotenv

from credit_score_mlops.modeling import WOELogisticRegression
from credit_score_mlops.plots import plot_calibration_curve
from credit_score_mlops.utils import read_yaml

app = typer.Typer()
load_dotenv(find_dotenv())


def get_or_create_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id


@app.command()
def main(train_file: Path, test_file: Path, model_file: Path) -> None:
    # 1. Load params:
    params = read_yaml(Path("params.yaml"))
    target = params["target"]
    log_reg_params = params["train"]["logistic_regression"]
    woe_transformer_params = params["train"]["weight_of_evidence_transformer"]
    mlflow_params = params["train"]["mlflow"]
    train_calibration_curve_file = params["train"]["train_calibration_curve_file"]
    test_calibration_curve_file = params["train"]["test_calibration_curve_file"]

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

    # 3. Track modelling experiment
    mlflow.set_tracking_uri(mlflow_params.remote_uri)  # set dagshub as the remote URI

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
        "DAGSHUB_USER_NAME"
    )  # set up credentials for accessing remote dagshub uri
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv(
        "DAGSHUB_PASSWORD"
    )  # set up credentials for accessing remote dagshub uri

    with mlflow.start_run(
        experiment_id=get_or_create_experiment_id(mlflow_params.experiment_name)
    ):
        # Model training
        model = WOELogisticRegression.from_parameters(
            woe_transformer_params=woe_transformer_params,
            logreg_params=log_reg_params,
        )
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        # Evaluations
        train_metrics = model.evaluate(y_train, y_train_pred_proba, "Training")
        test_metrics = model.evaluate(y_test, y_test_pred_proba, "Testing")

        # Log metrics
        for (train_metric, train_score), (test_metric, test_score) in zip(
            train_metrics.items(), test_metrics.items()
        ):
            mlflow.log_metric("Train {}".format(train_metric), train_score)
            mlflow.log_metric("Test {}".format(test_metric), test_score)

        # Log parameters
        mlflow.log_params(woe_transformer_params)
        mlflow.log_params(log_reg_params)

        # Log model
        mlflow.sklearn.log_model(model, mlflow_params.model_name)

        # Plot and log a calibration plot
        plot_calibration_curve(
            y_true=y_train,
            y_pred_proba=y_train_pred_proba,
            model_name=model.__class__.__name__,
            path=train_calibration_curve_file,
        )
        mlflow.log_artifact(train_calibration_curve_file)
        plot_calibration_curve(
            y_true=y_test,
            y_pred_proba=y_test_pred_proba,
            model_name=model.__class__.__name__,
            path=test_calibration_curve_file,
        )
        mlflow.log_artifact(test_calibration_curve_file)

        # End run
        mlflow.end_run()

    # 4. Save model and metrics locally
    model.save(model_file)


if __name__ == "__main__":
    typer.run(main)
