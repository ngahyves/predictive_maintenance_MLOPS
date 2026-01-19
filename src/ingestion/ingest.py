import requests
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("Ingestion")

#Class for loading errors
class DataLoadingError(Exception):
    pass
#Class for ingestion
class DataIngestor:
    def __init__(self, data_url: str, save_path: str):
        self.data_url = data_url
        self.save_path = Path(save_path)

    def download_data(self):
        try:
            logger.info(f"Downloading from: {self.data_url}")
            response = requests.get(self.data_url, timeout=10)
            response.raise_for_status()

            #Create the folder to save the file
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, "wb") as f:
                f.write(response.content)
            #Verify if it is empty
            if self.save_path.stat().st_size == 0:
                raise DataLoadingError("Downloaded file is empty.")

            logger.info("Download successful.")
        #Requests errors during downloading
        except requests.exceptions.Timeout:
            logger.error("Download timed out.")
            raise
        except requests.exceptions.ConnectionError:
            logger.error("Connection error during download.")
            raise
        except Exception as e:
            logger.error(f"Unexpected download error: {e}")
            raise

    #Function to load the data as a data frame
    def load_as_dataframe(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.save_path)
            return df
        # is the file empty?
        except pd.errors.EmptyDataError:
            logger.error(f"File is empty: {self.save_path}")
            raise DataLoadingError(f"File is empty: {self.save_path}")
        #Checking if the file is readable
        except pd.errors.ParserError:
            logger.error(f"Parsing error: {self.save_path}")
            raise DataLoadingError(f"Parsing error: {self.save_path}")
        #Checking another problem
        except Exception as e:
            logger.error(f"Unexpected error while loading data: {e}")
            raise DataLoadingError(f"Unexpected error: {e}")


if __name__ == "__main__":
    config = load_config()

    ingestor = DataIngestor(
        data_url=config["paths"]["data_url"],
        save_path=config["paths"]["save_path"]
    )

    ingestor.download_data()
    logger.info("Ingestion pipeline finished.")
