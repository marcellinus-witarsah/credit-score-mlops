# Credit Score MLOps
<p align="center">
    <img src="https://www.simmonsbank.com/siteassets/content-hub/learning-center/credit-score-image.jpg" alt="Credit Score Image" height="500">
    <p align="center">
        Figure 1: Credit Score Illustration (<a href="https://www.simmonsbank.com/siteassets/content-hub/learning-center/credit-score-image.jpg">Source</a>).
    </p>
</p>

## Project Summary
In this project, we developed a credit score model leveraging Logistic Regression and Weight of Evidence techniques. The scoring methodology is based on the "point to double the odds" approach, utilizing Logistic Regression parameters, Weight of Evidence, and specific user-defined constraints to assign credit points for based on each predictor variable. The development of credit score model are done manually without the help of `optbinning` (*like the previous one*).

## Project Scope
The main objective is not only to create a reliable credit score model and develop a comprehensive credit scorecard, but also emphasizes on model deployment through web application. Some of the concepts involve python **package development, continuous integration, and continuous deployment**.


## Tools and Technologies
The project is built using Python 3.10, with the following libraries and tools:
1. `pandas` and `numpy` for data manipulation.
2. `matplotlib` and `seaborn` for data visualization. 
3. `scikit-learn` for training and evaluation credit score model.
4. `fast-api` for API development.
5. `streamlit` for the development of the web application.
6. `azure cloud services` for hosting API services.

## Installation and Setup
To run this project locally, you can use [Anaconda](https://docs.anaconda.com/free/anaconda/install/). Ensure your Python version is 3.10. Recommended using **linux environment** for setting up  environment. Then, install the required libraries from the requirements.txt file:
```bash
  make create_environment  # create conda environment
  conda activate credit-score-mlops  # access the environment
  make requirements  # install all libraries from the requirements.txt file
  make create_ipykernel  # create ipykernel
```
With this you can use run the Python notebook using the exact same dependencies that I used for this project. For the web application you can access through this link https://huggingface.co/spaces/marcellinus-witarsah/credit-score-app-v2.
