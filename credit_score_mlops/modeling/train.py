"""
A module for model training.
"""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from credit_score_mlops.config import GLOBAL_PARAMS, MLFLOW_PARAMS, TRAIN_PARAMS
from credit_score_mlops.modeling import WOELogisticRegression
from credit_score_mlops.plots import plot_calibration_curve

load_dotenv(find_dotenv())


def get_or_create_experiment_id(name):
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        return exp_id
    return exp.experiment_id


def main():
    # 1. Load params:
    target = GLOBAL_PARAMS.target
    train_file = GLOBAL_PARAMS.train_file
    test_file = GLOBAL_PARAMS.test_file
    train_calibration_curve_file = GLOBAL_PARAMS.train_calibration_curve_file
    test_calibration_curve_file = GLOBAL_PARAMS.test_calibration_curve_file
    log_reg_params = TRAIN_PARAMS.logistic_regression
    woe_transformer_params = TRAIN_PARAMS.weight_of_evidence_transformer
    remote_uri = MLFLOW_PARAMS.remote_uri
    experiment_name = MLFLOW_PARAMS.experiment_name
    model_name = MLFLOW_PARAMS.model_name

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
    mlflow.set_tracking_uri(remote_uri)  # set dagshub as the remote URI

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
        "DAGSHUB_USER_NAME"
    )  # set up credentials for accessing remote dagshub uri
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv(
        "DAGSHUB_PASSWORD"
    )  # set up credentials for accessing remote dagshub uri

    with mlflow.start_run(experiment_id=get_or_create_experiment_id(experiment_name)):
        # 3.1 Model training
        model = WOELogisticRegression.from_parameters(
            woe_transformer_params=woe_transformer_params,
            logreg_params=log_reg_params,
        )
        model.fit(X_train, y_train)

        # 3.2 Predictions
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]

        # 3.3 Evaluations
        train_metrics = model.evaluate(y_train, y_train_pred_proba, "Training")
        test_metrics = model.evaluate(y_test, y_test_pred_proba, "Testing")

        # 3.4 Log Metrics
        for (train_metric, train_score), (test_metric, test_score) in zip(
            train_metrics.items(), test_metrics.items()
        ):
            mlflow.log_metric("Train {}".format(train_metric), train_score)
            mlflow.log_metric("Test {}".format(test_metric), test_score)

        # 3.5 Log Parameters
        mlflow.log_params(woe_transformer_params)
        mlflow.log_params(log_reg_params)

        # 3.6 Log model
        mlflow.sklearn.log_model(model, model_name)

        # 3.7 Plot and Log a Calibration Plot
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

        # 3.8 End Run
        mlflow.end_run()


if __name__ == "__main__":
    main()
