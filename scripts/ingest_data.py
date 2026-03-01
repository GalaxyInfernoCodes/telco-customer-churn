from telco_churn.data.ingestion import ingest_data

PROJECT_CONFIG_PATH = "config/project.yaml"

if __name__ == "__main__":
    ingest_data(PROJECT_CONFIG_PATH)
