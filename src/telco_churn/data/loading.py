import duckdb
import pandas as pd
from typing import Optional

def load_data_from_db(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Load data from a DuckDB table into a pandas DataFrame.
    
    Args:
        db_path (str): Path to the DuckDB database file.
        table_name (str): Name of the table to query.
        
    Returns:
        pd.DataFrame: The loaded data.
    """
    con = duckdb.connect(db_path)
    try:
        df = con.query(f"SELECT * FROM {table_name}").df()
        return df
    finally:
        con.close()
