import pandas as pd
import pandera as pa
from src.utils.logger import get_logger
from src.utils.config_loader import load_config

logger = get_logger("Validation")

class DataValidator:
    def __init__(self):
        # Defining the schema
        self.schema = pa.DataFrameSchema(
            columns={
                "UDI": pa.Column(int, pa.Check.gt(0)),
                "Product ID": pa.Column(str),
                "Type": pa.Column(str, pa.Check.isin(["L", "M", "H"])),
                "Air temperature [K]": pa.Column(float, pa.Check.between(280, 310)),
                "Process temperature [K]": pa.Column(float, pa.Check.between(280, 315)),
                "Rotational speed [rpm]": pa.Column(int, pa.Check.gt(0), coerce=True),
                "Torque [Nm]": pa.Column(float, pa.Check.gt(0)),
                "Tool wear [min]": pa.Column(int, pa.Check.ge(0), coerce=True),
                "Machine failure": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
                # Columns with failure
                "TWF": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
                "HDF": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
                "PWF": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
                "OSF": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
                "RNF": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
            },
            strict=True, #to Prohibit any column that is not defined 
            coerce=True #to convert automatically 
        )
    #Function to validate the contract
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Validation of the contract...")
        try:
            return self.schema.validate(df, lazy=True)# to see all the errors
        except pa.errors.SchemaErrors as err:
            logger.error(f"Validation failed :\n{err.failure_cases}")
            raise

if __name__ == "__main__":
    config = load_config()
    data_url = config["paths"]["data_url"]
    try:
        df = pd.read_csv(data_url)
        validator = DataValidator()
        validator.validate(df)
        logger.info("Validation passed.")
        print("✅ Validation")
    except Exception as e:
        logger.error(f"Error during validation : {e}")
        print("❌ Failure. Check “logs/pipeline.log” to see which column is causing the problem.")