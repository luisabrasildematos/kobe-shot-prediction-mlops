
import pandas as pd
import numpy as np
import mlflow
import os

from pathlib import Path
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import *

# define project base path robustly
project_root = Path(__file__).resolve().parent.parent.parent

# define paths
train_path = os.path.join(project_root, "Data", "Processed", "base_train.parquet")
test_path = os.path.join(project_root, "Data", "Processed", "base_test.parquet")
model_output_dir = os.path.join(project_root, "Data", "Modeling")

# ensure model output directory exists
os.makedirs(model_output_dir, exist_ok = True)

# set up mlflow
mlflow.set_experiment("Kobe_Shot_Prediction")

# start mlflow run
with mlflow.start_run(run_name = "Treinamento"):
    # load train and test datasets
    print(f"Loading training data from {train_path}...")
    train_df = pd.read_parquet(train_path)
    
    print(f"Loading test data from {test_path}")
    test_df = pd.read_parquet (test_path)

    # log dataset sizes
    mlflow.log_param("train_size", len(train_df))
    mlflow.log_param("test_size", len(test_df))

    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # initialize pycaret setup
    print("Initializing pycaret classification setup...")
    clf = setup(
        data = train_df,
        target = 'shot_made_flag',
        test_data = test_df,
        session_id = 42,
        log_experiment = True,
        experiment_name = "Kobe_Shot_Prediction",
        verbose = False
    )
    
    # train logistic regression model
    print("Training logistic regression model...")
    lr_model = create_model('lr', verbose = False)

    # generate predictions on test set
    lr_model = create_model('lr', verbose = False)

    #generate predictions on test set
    lr_predictions = predict_model(lr_model, data = test_df)
    lr_pred_proba = predict_model(lr_model, data = test_df, raw_score = True)

    print("Columns in lr_pred_proba:", lr_pred_proba.columns.tolist)

    #calculate metrics
    lr_logloss = log_loss(test_df['shot_made_flag'], lr_pred_proba['prediction_score_1'])
    lr_f1 = f1_score(test_df['shot_made_flag'], lr_predictions['prediction_label'])

    # log metrics to mlflow
    mlflow.log_metric("lr_log_loss", lr_logloss)
    mlflow.log_metric("lr_f1_score", lr_f1)

    print(f"Logistic regression - log loss: {lr_logloss:.4f}, F1 Score: {lr_f1:.4f}")

    # train decision tree model
    print("Training decision tree model...")
    dt_model = create_model('dt', verbose = False)

    # generate predictions on test set
    dt_predictions = predict_model(dt_model, data = test_df)
    dt_pred_proba = predict_model(dt_model, data = test_df, raw_score = True)

    # calculate metrics
    dt_logloss = log_loss(test_df['shot_made_flag'], dt_pred_proba['prediction_score_1'])
    dt_f1 = f1_score(test_df['shot_made_flag'], dt_predictions['prediction_label'])

    # log metrics to mlflow
    mlflow.log_metric("dt_log_loss", dt_logloss)
    mlflow.log_metric("dt_f1_score", dt_f1)

    print(f"Decision tree - log loss: {dt_logloss:.4f}, F1 Score: {dt_f1:.4f}")

    # model comparison and selection
    if dt_logloss < lr_logloss and dt_f1 > lr_f1:
        selected_model = dt_model
        model_name = "decision_tree"
        print("Selected model: Decision Tree (better log loss and F1 score)")
    elif dt_logloss < lr_logloss:
        selected_model = dt_model
        model_name = "decision_tree"
        print("Selected model: Decision tree (better log loss)")
    elif dt_f1 > lr_f1:
        selected_model = dt_model
        model_name = "decision_tree"
        print("Selected model: Decision tree (better F1 score)")
    else:
        selected_model = lr_model
        model_name = "logistic_regression"
        print("Selected model: logistic regression (better overall performance)")

    # log selected model details
    mlflow.log_param("selected_model", model_name)

    #save model
    saved_info = save_model(selected_model, model_name)
    print(f"PyCaret saved model to: {saved_info}")

    #log model via sklearn interface instead
    mlflow.sklearn.log_model(finalize_model(selected_model), model_name)

    print(f"Model training completed successfully")
    print(f"Selected model: {model_name}")