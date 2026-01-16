import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline # Pipeline qui supporte le SMOTE
from imblearn.over_sampling import SMOTE
from src.utils.logger import get_logger

class Preprocessor:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        self.target = config.get("target_col", "Machine failure")
        self.num_features = config.get("num_features", [])
        self.cat_features = config.get("cat_features", ["Type"])
        
        # On définit les briques de base de la pipeline
        self.full_pipeline = None

    def load_data(self) -> pd.DataFrame:
        self.logger.info(f"Chargement : {self.config['raw_data_path']}")
        return pd.read_csv(self.config["raw_data_path"])

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage : Doublons et colonnes inutiles."""
        self.logger.info("Nettoyage des données.")
        df = df.drop_duplicates()
        # On ignore les colonnes de pannes spécifiques pour ne pas tricher (Data Leakage)
        cols_to_drop = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
        return df

    def split_data(self, df: pd.DataFrame):
        """Split Train/Test."""
        self.logger.info("Split des données (Stratifié).")
        X = df.drop(columns=[self.target])
        y = df[self.target]
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def build_pipeline(self):
        """
        Définit la logique de transformation. 
        On utilise SimpleImputer pour garantir la robustesse.
        """
        self.logger.info("Construction de la Pipeline (Imputer + Scaler + Encoder).")
        
        # Pipeline Numérique
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Pipeline Catégorielle
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=[["L", "M", "H"]]))
        ])

        # ColumnTransformer pour assembler les deux
        preprocessor = ColumnTransformer([
            ("num", num_transformer, self.num_features),
            ("cat", cat_transformer, self.cat_features)
        ])

        # Pipeline finale avec SMOTE (uniquement pour le fit)
        self.full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42))
        ])

    def save_transformer(self):
        """Sauvegarde uniquement l'objet de transformation pour l'API."""
        path = Path(self.config["processor_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        # On ne sauvegarde que la partie 'preprocessor', pas le SMOTE !
        joblib.dump(self.full_pipeline.named_steps["preprocessor"], path)
        self.logger.info(f"Transformateur sauvegardé dans {path}")

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Sauvegarde les données finales prêtes pour l'entraînement."""
        path = Path(self.config["processed_data_dir"])
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump((X_train, X_test, y_train, y_test), path / "data_cleaned.joblib")
        self.logger.info("Données d'entraînement sauvegardées.")

    def run(self):
        """Orchestrateur conforme à ta demande."""
        self.logger.info("--- Démarrage Preprocessing ---")
        
        # 1. Chargement & Nettoyage
        df = self.load_data()
        df = self.clean_data(df)
        
        # 2. Split
        X_train, X_test, y_train, y_test = self.split_data(df)
        
        # 3. Build & Fit (avec SMOTE et Imputer)
        self.build_pipeline()
        
        self.logger.info("Application Fit & Resample (SMOTE)...")
        # fit_resample applique le preprocessor PUIS le SMOTE sur le train
        X_train_res, y_train_res = self.full_pipeline.fit_resample(X_train, y_train)
        
        # 4. Transformation du test (SANS SMOTE)
        X_test_transformed = self.full_pipeline.named_steps["preprocessor"].transform(X_test)
        
        # 5. Sauvegardes
        self.save_transformer()
        self.save_processed_data(X_train_res, X_test_transformed, y_train_res, y_test)
        
        self.logger.info("--- Preprocessing Terminé ---")
        return X_train_res, X_test_transformed, y_train_res, y_test

if __name__ == "__main__":
    conf = {
        "raw_data_path": "data/raw/ai4i2020.csv",
        "processed_data_dir": "data/processed",
        "processor_path": "models/preprocessor.joblib",
        "num_features": ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"],
        "cat_features": ["Type"]
    }
    logger = get_logger("Preprocessing")
    preprocessor = Preprocessor(conf, logger)
    preprocessor.run()