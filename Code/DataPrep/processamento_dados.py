import pandas as pd
import numpy as np
import mlflow
import os
from pathlib import Path

#configure mlflow
mlflow.set_experiment("Kobe_Shot_Prediction")

#define input and output paths
input_dev_path = "../../Data/Raw/dataset_kobe_dev.parquet"
input_prod_path = "../../Data/Raw/dataset_kobe_prod.parquet"
output_path = "../../Data/Processed/data_filtered.parquet"

#create output directory if doesn't exist still
os.makedirs(os.path.dirname(output_path), exist_ok = True)

#start mlflow run with specific name
with mlflow.start_run(run_name = "PreparacaoDados"):

    #log parameters
    mlflow.log_param("input_dev_path", input_dev_path)
    mlflow.log_param("input_prod_path", input_prod_path)
    mlflow.log_param("output_path", output_path)
    mlflow.log_param("remove_missing_values", True)

    #load delevopment data
    print(f"Loading data from {input_dev_path}...")
    df_dev = pd.read_parquet(input_dev_path)

    #check columns in dataset
    print("Columns available in dataset: ")
    print(df_dev.columns.tolist())

    #log initial dimensions
    initial_rows, initial_cols = df_dev.shape
    mlflow.log_metric("initial_rows", initial_rows)
    mlflow.log_metric("initial_cols", initial_cols)
    print(f"Initial dimensions: {initial_rows} rows, {initial_cols} columns")

    #select only specified columns
    columns_to_keep = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    print(f"Selecting columns: {columns_to_keep}")
    df_filtered = df_dev[columns_to_keep]

    #remove rows with missing values
    print("Removing rows with missing values")
    df_filtered = df_filtered.dropna()

    #log dimensions after filtering
    final_rows, final_cols = df_filtered.shape
    mlflow.log_metric("final_rows", final_rows)
    mlflow.log_metric("final_cols", final_cols)
    mlflow.log_metric("removed_rows", initial_rows - final_rows)

    print(f"Rows removed: {initial_rows - final_rows}")

    #save the processed dataset
    print(f"Saving processed dataset to {output_path}...")
    df_filtered.to_parquet(output_path)

    #log processed dataset as artifact in mlflow
    mlflow.log_artifact(output_path)

    print("Processing completed successfully")
    print(f"The resulting dataset dimension is: {final_rows} rows x {final_cols} columns")
