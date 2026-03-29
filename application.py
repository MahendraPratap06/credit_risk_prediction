from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from src.component.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=int(request.form.get('age')),
            annual_income=float(request.form.get('annual_income')),
            credit_score=int(request.form.get('credit_score')),
            loan_amount=float(request.form.get('loan_amount')),
            employment_years=int(request.form.get('employment_years')),
            existing_debt=float(request.form.get('existing_debt')),
            loan_term_years=int(request.form.get('loan_term_years'))
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Convert 0-1 score to percentage risk
        risk_score = round(float(results[0]) * 100, 2)

        if risk_score >= 70:
            risk_label = "High Risk"
        elif risk_score >= 40:
            risk_label = "Medium Risk"
        else:
            risk_label = "Low Risk"

        return render_template('home.html', results=risk_score, risk_label=risk_label)


if __name__ == "__main__":
    # BUG FIX: debug=True causes signal error on Python 3.13 Mac
    app.run(host="0.0.0.0", port=1080, debug=False, threaded=True, use_reloader=False)
