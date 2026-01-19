#Import the libaries
import joblib
import numpy as np
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    f1_score,
    classification_report,
    average_precision_score
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

#Calling the logger
logger = get_logger("Training")

#Model class trainer
class ModelTrainer:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        self.best_model_path = self.model_dir / "best_model.joblib"

        # MLflow experiment
        experiment_name = self.config["mlflow"]["experiment_name"]
        self.logger.info(f"Setting MLflow experiment: {experiment_name}")
        mlflow.set_experiment(experiment_name)

    #Load preprocessed data
    def load_processed_data(self):
        path = Path(self.config["paths"]["processed_data_dir"]) / "data_processed.joblib"
        self.logger.info(f"Loading processed data from: {path}")

        X_train, X_test, y_train, y_test = joblib.load(path)

        self.logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    #Building candidate models
    def get_models(self):
        self.logger.info("Building candidate models from config")

        cfg = self.config["models"]

        models = {
            "logreg": MultiOutputClassifier(
                LogisticRegression(**cfg["logreg"])
            ),
            "random_forest": MultiOutputClassifier(
                RandomForestClassifier(**cfg["random_forest"])
            ),
            "xgboost": MultiOutputClassifier(
                XGBClassifier(**cfg["xgboost"])
            )
        }

        return models
    #Choosing our metrics(Average precision micro/macro/by label)
    #Average precion is best suited for imbalanced data set
    def compute_average_precision(self, y_test, y_proba):
        ap_micro = average_precision_score(y_test, y_proba, average="micro")
        ap_macro = average_precision_score(y_test, y_proba, average="macro")

        ap_per_label = {}
        for i, col in enumerate(y_test.columns):
            ap_per_label[col] = average_precision_score(y_test.iloc[:, i], y_proba[:, i])

        return ap_micro, ap_macro, ap_per_label

    #Training
    def train_and_evaluate_model(self, name, model, params, X_train, X_test, y_train, y_test):
        self.logger.info(f"=== Training model: {name} ===")
        #Starting experiments with mlflow
        with mlflow.start_run(run_name=name):
            # Log hyperparams
            self.logger.info(f"[{name}] Hyperparameters: {params}")
            mlflow.log_params(params)
            mlflow.log_param("model_name", name)

            self.logger.info(f"[{name}] Fitting model...")
            model.fit(X_train, y_train)
            self.logger.info(f"[{name}] Model fitted successfully")

            # Probability for AP
            self.logger.info(f"[{name}] Predicting probabilities...")
            y_proba = model.predict_proba(X_test)
            y_proba = np.column_stack([p[:, 1] for p in y_proba])

            # AP metrics
            ap_micro, ap_macro, ap_per_label = self.compute_average_precision(y_test, y_proba)

            self.logger.info(f"[{name}] AP-micro: {ap_micro:.4f}")
            self.logger.info(f"[{name}] AP-macro: {ap_macro:.4f}")
            self.logger.info(f"[{name}] AP per label: {ap_per_label}")

            mlflow.log_metric("ap_micro", ap_micro)
            mlflow.log_metric("ap_macro", ap_macro)
            for label, score in ap_per_label.items():
                mlflow.log_metric(f"ap_{label}", score)

            # Making predictions
            y_pred = model.predict(X_test)

            # F1 metrics
            f1_micro = f1_score(y_test, y_pred, average="micro")
            f1_macro = f1_score(y_test, y_pred, average="macro")

            self.logger.info(f"[{name}] F1-micro: {f1_micro:.4f}")
            self.logger.info(f"[{name}] F1-macro: {f1_macro:.4f}")

            mlflow.log_metric("f1_micro", f1_micro)
            mlflow.log_metric("f1_macro", f1_macro)

            # Classification report
            report = classification_report(y_test, y_pred)
            self.logger.info(f"[{name}] Classification report:\n{report}")

            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")

        # AP-micro as main metric
        return model, ap_micro

    def train(self):
        self.logger.info("=== Starting training of multiple models ===")

        X_train, X_test, y_train, y_test = self.load_processed_data()
        models = self.get_models()
        cfg_models = self.config["models"]

        best_model = None
        best_score = -1
        best_name = None

        for name, model in models.items():
            params = cfg_models[name]
            trained_model, score = self.train_and_evaluate_model(
                name, model, params, X_train, X_test, y_train, y_test
            )

            if score > best_score:
                self.logger.info(f"New best model: {name} with AP-micro={score:.4f}")
                best_score = score
                best_model = trained_model
                best_name = name

        self.logger.info(f"Best model is: {best_name} with AP-micro={best_score:.4f}")
        self.logger.info(f"Saving best model to: {self.best_model_path}")
        joblib.dump(best_model, self.best_model_path)

        self.logger.info("=== Training of all models completed ===")

        return best_model, best_name, best_score


if __name__ == "__main__":
    config = load_config()
    trainer = ModelTrainer(config, logger)
    trainer.train()
