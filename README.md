# Credit Risk Predictor

ML project to predict loan default risk score using regression.

## Setup

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python src/component/data_ingestion.py
```

## Run the app

```bash
python application.py
```

Open `http://localhost:1080`

## Project Structure

```
credit_risk/
├── src/
│   ├── component/
│   │   ├── data_ingestion.py
│   │   ├── data_trasformation.py
│   │   ├── model_trainer.py
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── index.html
│   └── home.html
├── artifacts/          # generated after training
├── logs/               # generated at runtime
├── credit_risk_prediction_1500.csv
├── application.py
├── setup.py
├── requirements.txt
└── Dockerfile
```
