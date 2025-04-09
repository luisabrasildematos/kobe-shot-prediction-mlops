
import pandas as pd
import numpy as np
import mlflow
import os
import shutil
from pathlib import Path
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import create_model, setup, predict_model, save_model

#define project base path robustly
project_root = Path(__file__).resolve().parent.parent.parent

#define paths
prod_data_path = os.path.join(project_root, "Data", "Raw", "dataset_kobe_prod.parquet")
train_path = os.path.join(project_root, "Data", "Processed", "base_train.parquet")
results_path = os.path.join(project_root, "Data", "Processed", "production_result.parquet")
model_path = os.path.join(project_root, "Data", "Modeling", "decision_tree.pkl")
artifacts_dir = os.path.join(project_root, "mlruns", "artifacts")

#ensure artifacts directory exists
os.makedirs(artifacts_dir, exist_ok=True)

#set up mlflow
mlflow.set_experiment("Kobe_Shot_Prediction")

#start mlflow run
with mlflow.start_run(run_name="PipelineAplicacao"):
   #log parameters
   mlflow.log_param("prod_data_path", prod_data_path)
   mlflow.log_param("results_path", results_path)

   #load production data
   print(f"Loading production data from {prod_data_path}...")
   prod_df = pd.read_parquet(prod_data_path)

   #load training data
   print(f"Loading training data from {train_path}...")
   train_df = pd.read_parquet(train_path)

   #initial data size
   initial_rows = len(prod_df)
   mlflow.log_metric("initial_rows", initial_rows)
   print(f"Production dataset size: {initial_rows} rows")

   #filter columns to match training data
   columns_to_keep = ['lat', 'lon', 'minutes_remaining', 'period', 
                      'playoffs', 'shot_distance', 'shot_made_flag']

   prod_df = prod_df[columns_to_keep]
   print(f"Filtered production data to {len(columns_to_keep)} columns")

   #remove rows with missing values
   prod_df = prod_df.dropna()
   rows_removed = initial_rows - len(prod_df)
   mlflow.log_metric("rows_removed", rows_removed)
   print(f"Removed {rows_removed} rows with missing values")
   print(f"Final production dataset size: {len(prod_df)} rows")

   #setup pycaret for training data
   print("Setting up PyCaret environment for training data...")
   clf = setup(data=train_df, target='shot_made_flag', session_id=42, verbose=False)

   #train decision tree model
   print("Training a decision tree model...")
   dt_model = create_model('dt', verbose=False)

   #apply model to production data
   print("Applying model to production data...")
   predictions = predict_model(dt_model, data=prod_df, raw_score=True)

   #calculate metrics
   prod_logloss = log_loss(prod_df['shot_made_flag'], predictions['prediction_score_1'])
   prod_f1 = f1_score(prod_df['shot_made_flag'], predictions['prediction_label'])

   #log metrics
   mlflow.log_metrics({
       "prod_log_loss": prod_logloss,
       "prod_f1_score": prod_f1
   })

   print(f"Production metrics - log loss: {prod_logloss:.4f}, F1 Score: {prod_f1:.4f}")

   #prepare results
   results_df = predictions[['prediction_label', 'prediction_score_1']].copy()
   results_df['actual'] = prod_df['shot_made_flag']
   results_df.to_parquet(results_path)

   #log artifacts
   temp_artifacts_dir = os.path.join(artifacts_dir, "production_results")
   os.makedirs(temp_artifacts_dir, exist_ok=True)
   
   #copy results file to artifacts directory
   results_artifact_path = os.path.join(temp_artifacts_dir, "production_result.parquet")
   shutil.copy(results_path, results_artifact_path)
   
   #log the artifacts directory
   mlflow.log_artifacts(temp_artifacts_dir)

   #save model
   model_save_path = os.path.join(project_root, "Data", "Modeling", "decision_tree.pkl")
   os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
   save_model(dt_model, model_save_path)

   print(f"Model saved to: {model_save_path}")
   print("\nPipeline application completed successfully")