import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("Prediction")


class Predictor:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

        # Paths
        self.model_path = Path(self.config["paths"]["model_dir"]) / "best_model.joblib"
        self.preprocessor_path = Path(self.config["paths"]["processed_data_dir"]) / "preprocessor.joblib"

        # Load model
        self.logger.info(f"Loading best model from: {self.model_path}")
        try:
            self.model = joblib.load(self.model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

        # Load preprocessing pipeline
        self.logger.info(f"Loading preprocessor from: {self.preprocessor_path}")
        try:
            self.preprocessor = joblib.load(self.preprocessor_path)
            self.logger.info("Preprocessor loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load preprocessor: {e}")
            raise

        # Load label names
        self.label_columns = self.config["data"]["label_columns"]
        self.logger.info(f"Loaded label columns: {self.label_columns}")

    def preprocess_input(self, input_data: dict) -> np.ndarray:
        """
        Convert raw input into a dataframe and apply preprocessing.
        """
        self.logger.info("Starting preprocessing of input data")

        try:
            df = pd.DataFrame([input_data])
            self.logger.debug(f"Raw input converted to DataFrame: {df.to_dict(orient='records')}")
            X = self.preprocessor.transform(df)
            self.logger.info("Preprocessing completed successfully")
            return X
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {e}")
            raise

    def predict(self, input_data: dict) -> dict:
        """
        Predict multilabel outputs for a single input.
        """
        self.logger.info("Starting prediction process")
        self.logger.debug(f"Received input data: {input_data}")

        # Preprocess
        X = self.preprocess_input(input_data)

        # Predict probabilities
        try:
            self.logger.info("Predicting probabilities...")
            y_proba = self.model.predict_proba(X)
            y_proba = np.column_stack([p[:, 1] for p in y_proba])
            self.logger.debug(f"Predicted probabilities: {y_proba.tolist()}")
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {e}")
            raise

        # Predict labels
        try:
            self.logger.info("Converting probabilities to binary predictions (threshold=0.5)")
            y_pred = (y_proba >= 0.5).astype(int)
            self.logger.debug(f"Binary predictions: {y_pred.tolist()}")
        except Exception as e:
            self.logger.error(f"Error during label prediction: {e}")
            raise

        # Format output
        predictions = {
            label: int(y_pred[0][i])
            for i, label in enumerate(self.label_columns)
        }

        probabilities = {
            label: float(y_proba[0][i])
            for i, label in enumerate(self.label_columns)
        }

        self.logger.info(f"Prediction completed successfully: {predictions}")

        return {
            "predictions": predictions,
            "probabilities": probabilities
        }


if __name__ == "__main__":
    config = load_config()
    predictor = Predictor(config, logger)

    # Example input
    sample = {
        "Air temperature [K]": 300,
        "Process temperature [K]": 310,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40,
        "Tool wear [min]": 100,
        "Type": "L"
    }

    result = predictor.predict(sample)
    print(result)
