import duckdb
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from telco_churn.config import ProjectConfig, load_config

def ingest_data(project_config_path: str):
    """
    Ingest data from CSV, split into train/val/test, and save to DuckDB.
    
    Args:
        project_config_path (str): Path to the project configuration file.
    """
    # Load project config
    project_config = load_config(project_config_path, ProjectConfig)
    
    # Paths
    csv_path = os.path.join(project_config.paths["artifacts"], "Telco-Customer-Churn.csv")
    db_path = project_config.db_path
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}. Please ensure the data file exists.")
        return

    # Split data
    # 60% Train, 20% Val, 20% Test
    train_val, test = train_test_split(df, test_size=0.2, random_state=project_config.seed)
    train, val = train_test_split(train_val, test_size=0.25, random_state=project_config.seed) # 0.25 * 0.8 = 0.2
    
    print(f"Data split shapes:")
    print(f"Train: {train.shape}")
    print(f"Validation: {val.shape}")
    print(f"Test: {test.shape}")
    
    # Connect to DuckDB
    con = duckdb.connect(db_path)
    
    # Write to tables
    print("Writing to DuckDB...")
    con.execute(f"CREATE OR REPLACE TABLE {project_config.tables['train']} AS SELECT * FROM train")
    con.execute(f"CREATE OR REPLACE TABLE {project_config.tables['validation']} AS SELECT * FROM val")
    con.execute(f"CREATE OR REPLACE TABLE {project_config.tables['test']} AS SELECT * FROM test")
    
    # Verify
    print("Tables in DuckDB:")
    print(con.execute("SHOW TABLES").fetchall())
    
    con.close()
    print(f"Ingestion complete. Database saved to {db_path}")
