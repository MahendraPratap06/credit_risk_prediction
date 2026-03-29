import sys
import os
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            # Clamp to [0,1] since regression can predict outside this range
            preds = np.clip(preds, 0, 1)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        age: int,
        annual_income: float,
        credit_score: int,
        loan_amount: float,
        employment_years: int,
        existing_debt: float,
        loan_term_years: int
    ):
        self.age = age
        self.annual_income = annual_income
        self.credit_score = credit_score
        self.loan_amount = loan_amount
        self.employment_years = employment_years
        self.existing_debt = existing_debt
        self.loan_term_years = loan_term_years

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "annual_income": [self.annual_income],
                "credit_score": [self.credit_score],
                "loan_amount": [self.loan_amount],
                "employment_years": [self.employment_years],
                "existing_debt": [self.existing_debt],
                "loan_term_years": [self.loan_term_years]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
