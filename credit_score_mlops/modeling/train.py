"""
A module for model training.
"""

import pandas as pd
import typer
from pathlib import Path
from credit_score_mlops.config import MODELS_DIR, PROCESSED_DATA_DIR
from credit_score_mlops.modeling import WOELogisticRegression
import dvc.api

app = typer.Typer()


@app.command()
def main(
    train_file: Path = PROCESSED_DATA_DIR / "train.csv",
    model_file: Path = MODELS_DIR / "model.pkl",
):
    # 1. Load params:
    params = dvc.api.params_show()

    target = params["target"]
    woe_transformer_params = params["train"]["woe_transformer_params"]
    logreg_params = params["train"]["logreg_params"]

    # 2. Load data
    train_df = pd.read_csv(train_file)
    X_train, y_train = (
        train_df.drop(columns=[target]),
        train_df[target],
    )

    # 3. Initialize model
    model = WOELogisticRegression.from_parameters(
        woe_transformer_params=woe_transformer_params,
        logreg_params=logreg_params,
    )

    # 4. Train model
    model.fit(X_train, y_train)

    # 5. Save model
    model.save(model_file)


if __name__ == "__main__":
    main()
