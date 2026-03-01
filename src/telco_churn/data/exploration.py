import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from telco_churn.config import ProjectConfig, load_config

def run_exploration(project_config_path: str):
    """
    Perform exploratory data analysis and save plots.
    """
    project_config = load_config(project_config_path, ProjectConfig)
    
    # Paths
    csv_path = os.path.join(project_config.paths["artifacts"], "Telco-Customer-Churn.csv")
    output_dir = os.path.join(project_config.paths["artifacts"], "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {csv_path} for exploration...")
    try:
        telco_full = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {csv_path}. Please run ingestion first.")
        return

    print("First 5 rows:")
    print(telco_full.head())

    print("\nDataset Info:")
    print(telco_full.info())

    print("\nDescriptive Statistics:")
    print(telco_full.describe())

    # Save individual plots
    columns = telco_full.columns
    print(f"Generating histograms for {len(columns)} columns...")
    for column in columns:
        plt.figure(figsize=(10, 4))
        sns.histplot(telco_full[column])
        plt.title(f'Distribution of {column}')
        plt.savefig(f'{output_dir}/{column}_hist.png')
        plt.close()

    # Create combined plot
    num_cols = len(columns)
    num_rows = math.ceil(num_cols / 3)

    fig, axes = plt.subplots(num_rows, 3, figsize=(20, num_rows * 5))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        sns.histplot(telco_full[column], ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')

    # Remove empty subplots
    for i in range(num_cols, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    combined_path = os.path.join(output_dir, "combined_histograms.png")
    plt.savefig(combined_path)
    plt.close()
    print(f"Exploration complete. Plots saved to {output_dir}")
