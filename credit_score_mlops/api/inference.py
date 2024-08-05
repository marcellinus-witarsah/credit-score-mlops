"""
a module for data preprocessing.
"""

import os

import mlflow.sklearn
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from credit_score_mlops.config import (
    CREDIT_SCORE_SCALING_PARAMS,
    GLOBAL_API,
    MLFLOW_PARAMS,
)
from credit_score_mlops.credit_score import CreditScoreScaling

load_dotenv(find_dotenv())

app = FastAPI()


class LoanApplicantData(BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


class CreditScore(BaseModel):
    credit_score: int


def prepare_credit_scorer():
    # Credit Scorer
    logged_model = GLOBAL_API.logged_model
    remote_uri = MLFLOW_PARAMS.remote_uri
    pdo = CREDIT_SCORE_SCALING_PARAMS.pdo
    odds = CREDIT_SCORE_SCALING_PARAMS.odds
    scorecard_points = CREDIT_SCORE_SCALING_PARAMS.scorecard_points

    # 2. Access remote MLFlow Server on DagsHub
    mlflow.set_tracking_uri(remote_uri)  # set dagshub as the remote URI

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv(
        "DAGSHUB_USER_NAME"
    )  # set up credentials for accessing remote dagshub uri
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv(
        "DAGSHUB_PASSWORD"
    )  # set up credentials for accessing remote dagshub uri

    # 3. Get Model for Remote URI
    loaded_model = mlflow.sklearn.load_model(logged_model)

    # 4. Integrate with the CreditScoreScaling class
    credit_scorer = CreditScoreScaling(
        pipeline=loaded_model.pipeline, pdo=pdo, odds=odds, scorecard_points=scorecard_points
    )
    return credit_scorer


credit_scorer = prepare_credit_scorer()


@app.get("/")
async def info() -> dict:
    """Calculate credit score based on the loan applicant data.

    Args:
        loan_applicant_data (LoanApplicantData): Load applicant data.

    Returns:
        CreditScore: Credits score results returned in dictionary format.
    """
    return {"message": "Welcome to Credit Score API!"}


@app.post("/calculate-credit-score")
async def calculate_credit_score(loan_applicant_data: LoanApplicantData) -> CreditScore:
    """Calculate credit score based on the loan applicant data.

    Args:
        loan_applicant_data (LoanApplicantData): Load applicant data.

    Returns:
        CreditScore: Credits score results returned in dictionary format.
    """
    input_df = pd.DataFrame(
        [loan_applicant_data.model_dump().values()],
        columns=loan_applicant_data.model_dump().keys(),
    )
    credit_score = credit_scorer.calculate_credit_score(input_df)["credit_score"][0]
    return {"credit_score": round(credit_score)}
