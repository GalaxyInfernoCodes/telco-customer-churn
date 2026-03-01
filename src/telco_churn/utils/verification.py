import os
import duckdb
import mlflow
import pandas as pd
from typing import Optional
from telco_churn.config import ProjectConfig, load_config

def verify_data_setup(project_config_path: str = "config/project.yaml") -> bool:
    """
    Verify that the data setup is correct:
    - Project configuration file exists.
    - DuckDB database file exists.
    - Required tables (train, validation, test) exist in DuckDB.
    """
    if not os.path.exists(project_config_path):
        print(f"Project config not found at: {project_config_path}")
        return False
        
    project_config = load_config(project_config_path, ProjectConfig)
    db_path = project_config.db_path
    
    if not os.path.exists(db_path):
        print(f"DuckDB database not found at: {db_path}")
        return False
        
    print(f"DuckDB found at: {db_path}")
    con = duckdb.connect(db_path)
    try:
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        required_tables = list(project_config.tables.values())
        
        for table in required_tables:
            if table not in tables:
                print(f"Required table '{table}' missing from DuckDB.")
                return False
            
            count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print(f"Table '{table}' contains {count} rows.")
            
        print("Data setup verification passed.")
        return True
    except Exception as e:
        print(f"Verification failed with error: {e}")
        return False
    finally:
        con.close()

def verify_model_inference(project_config_path: str = "config/project.yaml", experiment_name: Optional[str] = None) -> bool:
    """
    Verify that the latest model in MLflow can be loaded and perform inference.
    """
    project_config = load_config(project_config_path, ProjectConfig)
    exp_name = experiment_name or project_config.experiment_name or "telco_churn_wrapper"
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(exp_name)
    
    runs = mlflow.search_runs(order_by=["start_time desc"], max_results=1)
    if runs.empty:
        print(f"No runs found in experiment: {exp_name}")
        return False
        
    last_run_id = runs.iloc[0]["run_id"]
    print(f"Verifying model from run: {last_run_id}")
    
    model_uri = f"runs:/{last_run_id}/model"
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False

    # Load small sample for prediction
    con = duckdb.connect(project_config.db_path)
    try:
        test_df = con.query(f"SELECT * FROM {project_config.tables['test']} LIMIT 5").df()
    finally:
        con.close()
    
    if test_df.empty:
        print("No test data found for inference check.")
        return False
        
    try:
        predictions = loaded_model.predict(test_df)
        print(f"Inference check passed. Predictions: {predictions}")
        return True
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False
