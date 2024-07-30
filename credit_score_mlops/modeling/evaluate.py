"""
A module for model evaluation.
"""

import pandas as pd

from credit_score_mlops.config import EVALUATE_CONFIG
from credit_score_mlops.modeling import WOELogisticRegression
from credit_score_mlops.utils import save_json
from credit_score_mlops.plots import plot_calibration_curve


def predict():

    # 1. Load data
    train_df = pd.read_csv(EVALUATE_CONFIG.train_file)
    test_df = pd.read_csv(EVALUATE_CONFIG.test_file)
    X_train, y_train = (
        train_df.drop(columns=[EVALUATE_CONFIG.target]),
        train_df[EVALUATE_CONFIG.target],
    )

    X_test, y_test = (
        test_df.drop(columns=[EVALUATE_CONFIG.target]),
        test_df[EVALUATE_CONFIG.target],
    )

    # 2. Initialize model
    model = WOELogisticRegression.from_file(EVALUATE_CONFIG.model_file)

    # 3. Evaluate model performance
    train_eval_results = model.evaluate(X_train, y_train, "Training")
    test_eval_results = model.evaluate(X_test, y_test, "Testing")

    # 4. Save results
    save_json(data=train_eval_results, path=EVALUATE_CONFIG.train_metrics_file)
    plot_calibration_curve(
        y_true=y_train,
        y_pred_proba=model.predict_proba(X_train)[:, 1],
        model_name=model.__class__.__name__,
        path=EVALUATE_CONFIG.train_calibration_curve_file,
    )
    save_json(data=test_eval_results, path=EVALUATE_CONFIG.test_metrics_file)
    plot_calibration_curve(
        y_true=y_test,
        y_pred_proba=model.predict_proba(X_test)[:, 1],
        model_name=model.__class__.__name__,
        path=EVALUATE_CONFIG.test_calibration_curve_file,
    )


if __name__ == "__main__":
    predict()
