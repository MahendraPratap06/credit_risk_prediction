import os, sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:,  :-1], test_arr[:,  -1]

            models = {
                "Logistic Regression":      LogisticRegression(max_iter=1000),
                "Decision Tree":            DecisionTreeClassifier(),
                "Random Forest":            RandomForestClassifier(),
                "Gradient Boosting":        GradientBoostingClassifier(),
                "XGB Classifier":           XGBClassifier(eval_metric="logloss"),
                "CatBoost Classifier":      CatBoostClassifier(verbose=False),
                "AdaBoost Classifier":      AdaBoostClassifier(),
                "KNN Classifier":           KNeighborsClassifier(),
            }
            params = {
                "Logistic Regression":  {"C": [0.1, 1.0, 10.0]},
                "Decision Tree":        {"max_depth": [3, 5, 10]},
                "Random Forest":        {"n_estimators": [50, 100, 200]},
                "Gradient Boosting":    {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                "XGB Classifier":       {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]},
                "CatBoost Classifier":  {"depth": [4, 6, 8], "learning_rate": [0.05, 0.1]},
                "AdaBoost Classifier":  {"n_estimators": [50, 100]},
                "KNN Classifier":       {"n_neighbors": [3, 5, 7]},
            }

            model_report = evaluate_model(X_train, y_train, X_test, y_test, models, params)

            best_name  = max(model_report, key=model_report.get)
            best_score = model_report[best_name]
            best_model = models[best_name]

            if best_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model: {best_name}  AUC-ROC: {best_score:.4f}")
            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            y_proba = best_model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            return auc, best_name

        except Exception as e:
            raise CustomException(e, sys)