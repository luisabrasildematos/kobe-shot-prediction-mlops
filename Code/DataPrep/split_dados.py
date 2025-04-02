import pandas as pd
import mlflow
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

#define project base path 
project_root = Path(__file__).resolve().parent.parent.parent

#define paths
input_path = os.path.join(project_root, "Data", "Processed", "data_filtered.parquet")
train_output_path = os.path.join(project_root, "Data", "Processed", "base_train.parquet")
test_output_path = os.path.join(project_root, "Data", "Processed", "base_test.parquet")

#ensure output directory exists
os.makedirs(os.path.dirname(train_output_path), exist_ok=True)

#set up mlflow
mlflow.set_experiment("Kobe_Shot_Prediction")

#start mlflow run
with mlflow.start_run(run_name = "DataSplit"):
    #log parameters
    test_size = 0.2
    random_state = 42
    stratify = True

    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("stratify", stratify)

    #load filtered dataset
    print(f"Loading filtered data from {input_path}...")
    df = pd.read_parquet(input_path)

    #log initial data size
    total_rows = len(df)
    mlflow.log_metric("total_rows", total_rows)
    print(f"Total dataset size: {total_rows} rows")

    #split data into training and testing sets
    X = df.drop('shot_made_flag', axis = 1)
    y = df['shot_made_flag']

    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size, random_state = random_state, stratify = y
        )
        print(f"Using stratified split based on target variable")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size = test_size, random_state = random_state
        )
    
    #combine features and targets back into dataframes
    train_df = pd.concat([X_train, y_train], axis = 1)
    test_df = pd.concat([X_test, y_test], axis = 1)

    #log split metrics
    train_rows = len(train_df)
    test_rows = len(test_df)
    mlflow.log_metric("train_rows", train_rows)
    mlflow.log_metric("test_rows", test_rows)

    #calculate class distribution in train and test sets
    train_positive_rate = train_df['shot_made_flag'].mean()
    test_positive_rate = test_df['shot_made_flag'].mean()
    mlflow.log_metric("train_positive_rate", train_positive_rate)
    mlflow.log_metric("test_positive_rate", test_positive_rate)

    print(f"Train set: {train_rows} rows ({train_rows/total_rows:.1%})")
    print(f"Test set: {test_rows} rows ({test_rows/total_rows:.1%})")
    print(f"Train set positive rate: {train_positive_rate:.2%}")
    print(f"Test set positive rate: {test_positive_rate:.2%}")

    #save datasets to parquet files
    print(f"Saving train dataset to {train_output_path}")
    train_df.to_parquet(train_output_path)

    print(f"Saving test dataset to {test_output_path}")
    test_df.to_parquet(test_output_path)

    #log artifacts
    mlflow.log_artifact(train_output_path)
    mlflow.log_artifact(test_output_path)

    print("Data split completed successfully.")
    print(f"Train dataset: {train_rows} rows")
    print(f"Test dataset: {test_rows} rows")

