import requests
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("Ingestion")

class DataIngestor:
    def __init__(self, data_url: str, save_path: str):
        self.data_url = data_url
        self.save_path = Path(save_path)

    def download_data(self):
        try:
            logger.info(f"Download from : {self.data_url}")
            response = requests.get(self.data_url, timeout=10)
            response.raise_for_status()
            
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, 'wb') as f:
                f.write(response.content)
            logger.info("Download successful.")
        except Exception as e:
            logger.error(f"Download error : {e}")
            raise

    def load_as_dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.save_path)

if __name__ == "__main__":
    config = load_config()
    # Using the config file informations
    ingestor = DataIngestor(
        data_url=config["paths"]["data_url"], # On prend la clé exacte du YAML
        save_path=config["paths"]["raw_data_path"]
    )
    
    ingestor.download_data()
    logger.info("Pipeline d'ingestion terminé.")