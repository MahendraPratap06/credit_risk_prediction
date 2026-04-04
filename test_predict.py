from src.component.predict_pipeline import CustomData, PredictPipeline
import pandas as pd

try:
    data = CustomData(
        age=25,
        annual_income=50000.0,
        credit_score=700,
        loan_amount=10000.0,
        employment_years=3,
        existing_debt=2000.0,
        loan_term_years=5
    )
    pred_df = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    print(predict_pipeline.predict(pred_df))
except Exception as e:
    print("Error:", e)
    import traceback
    traceback.print_exc()
