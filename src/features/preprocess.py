#Import libaries
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
#Calling the logger
logger = get_logger("Preprocessing")

#Preprocessing pipeline class
class Preprocessor:
    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

        self.logger.info("Initializing Preprocessor...")

        # MULTILABEL TARGETS
        self.target_cols = self.config["features"]["target_cols"]
        self.logger.info(f"Target columns: {self.target_cols}")

        self.num_features = self.config["features"]["num_features"]
        self.cat_features = self.config["features"]["cat_features"]

        self.logger.info(f"Numerical features: {self.num_features}")
        self.logger.info(f"Categorical features: {self.cat_features}")

        self.processor_path = Path(self.config["paths"]["processor_path"])
        self.processed_dir = Path(self.config["paths"]["processed_data_dir"])

        self.preprocessor = None

    def load_data(self):
        path = self.config["paths"]["save_path"]
        self.logger.info(f"Loading raw data from: {path}")
        df = pd.read_csv(path)
        self.logger.info(f"Raw data loaded. Shape: {df.shape}")
        return df

    def clean_data(self, df):
        self.logger.info("Cleaning data: removing duplicates and unused columns")
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        self.logger.info(f"Removed {before - after} duplicates")

        df = df.drop(columns=["UDI", "Product ID", "Machine failure"], errors="ignore")
        self.logger.info(f"Remaining columns: {list(df.columns)}")
        return df

    def split_data(self, df):
        self.logger.info("Splitting data into train/test sets (multilabel)")
        X = df.drop(columns=self.target_cols)
        y = df[self.target_cols]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def build_pipeline(self):
        self.logger.info("Building preprocessing pipeline...")

        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", num_transformer, self.num_features),
            ("cat", cat_transformer, self.cat_features)
        ])

        self.logger.info("Preprocessing pipeline successfully built.")

    def run(self):
        self.logger.info("=== Starting multilabel preprocessing ===")

        df = self.load_data()
        df = self.clean_data(df)

        X_train, X_test, y_train, y_test = self.split_data(df)

        self.build_pipeline()

        self.logger.info("Fitting preprocessor on training data...")
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        self.logger.info(f"Training data transformed. Shape: {X_train_transformed.shape}")

        self.logger.info("Transforming test data...")
        X_test_transformed = self.preprocessor.transform(X_test)
        self.logger.info(f"Test data transformed. Shape: {X_test_transformed.shape}")

        # Save preprocessor
        self.logger.info(f"Saving preprocessor to: {self.processor_path}")
        self.processor_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor, self.processor_path)

        # Save processed data
        self.logger.info(f"Saving processed datasets to: {self.processed_dir}")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            (X_train_transformed, X_test_transformed, y_train, y_test),
            self.processed_dir / "data_processed.joblib"
        )

        self.logger.info("=== Preprocessing completed successfully ===")

        return X_train_transformed, X_test_transformed, y_train, y_test

if __name__ == "__main__":
    config = load_config()
    preprocessor = Preprocessor(config, logger)
    preprocessor.run()

