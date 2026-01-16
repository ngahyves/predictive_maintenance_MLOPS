import requests
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("Ingestion")

class DataIngestor:
    def __init__(self, source_url: str, save_path: str):
        self.source_url = source_url
        self.save_path = Path(save_path)

    def download_data(self):
        """Download the file from URL."""
        logger.info(f"Download file from: {self.source_url}")
        try:
            response = requests.get(self.source_url)
            response.raise_for_status()  # Check HTTP 200
            
            # Create the data/raw folder if it does not exist
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the raw data
            with open(self.save_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"File downloaded and saved in: {self.save_path}")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during download: {e}")
            raise ConnectionError(f"Unable to connect to the server: {e}")

    def load_as_dataframe(self) -> pd.DataFrame:
        """Load the downloaded file into a DataFrame."""
        try:
            df = pd.read_csv(self.save_path)
            logger.info(f"Data loaded into memory. Size: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            raise


# -------------------------
# TEST
# -------------------------

if __name__ == "__main__":
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
    SAVE_TO = "data/raw/ai4i2020.csv"

    ingestor = DataIngestor(URL, SAVE_TO)
    ingestor.download_data()
    data = ingestor.load_as_dataframe()

    print("\n--- test passed ---")
    print(data.head())
