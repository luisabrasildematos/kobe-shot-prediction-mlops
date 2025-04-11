
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Define project base path
project_root = Path(__file__).resolve().parent.parent.parent

# Define paths
train_path = os.path.join(project_root, "Data", "Processed", "base_train.parquet")
test_path = os.path.join("Data", "Processed", "base_test.parquet")
prod_results_path = os.path.join("project_root", "Data", "Processed", "production_results.parquet")

# Set up page
st.title("Kobe Bryant Shot Prediction Monitor")
st.markdown('<p style = "font-size:48px;">🏀</p>',
unsafe_allow_html = True)

# Load data
train_df = pd.read_parquet(train_path) if os.path.exists(train_path) else None
test_df = pd.read_parquet(test_path) if os.path.exists(test_path) else None
prod_results = pd.read_parquet(prod_results_path) if os.path.exists(prod_results_path) else None

# Performance metrics section
st.header("1. Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Performance")
    if train_df is not None:
        st.metric("Training samples", f"{len(train_df):,}")
        st.metric("Shot success rate", f"{train_df['shot_made_flag'].mean()*100:.1f}%")

        # Training distribution
        fig, ax = plt.subplots()
        train_df['shot_made_flag'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Shot Distribution - Training")
        st.pyplot(fig)

with col2:
    st.subheader("Production Performance")
    if prod_results is not None:
        st.metric("Production samples", f"{len(prod_results):,}")
        st.metric("Prediction accuracy", f"{(prod_results['actual'] == prod_results['prediction_label']).mean()*100:.1f}%")

        # Confusion matrix
        fig, ax = plt.subplots()
        cm = confusion_matrix(prod_results['actual'], prod_results['prediction_label'])
        ax.imshow(cm, cmap='Blues')
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        st.pyplot(fig)

# Model health monitoring
st.header("2. Model Health Monitoring")
st.write("""
### Monitoring with Shot_Made_Flag available:
- Calculate F1 score and log loss on new data
- Compare against training performance
- Set alert thresholds for degradation

### Monitoring without Shot_Made_Flag:
- Track features distribution vs. training data
- Monitor prediction distribution changes
- Track confidence score of predictions 
""")

# Retraining strategies
st.header("3. Retraining Strategies")

st.subheader("Reactivate Strategy")
st.write("Triggers model updates based on performance degradation:")
st.write("- Set performance thresholds (F1 score, log loss)")
st.write("- When metrics cross threshold, initiate retraining")
st.write("- Compare new model against current production model")

st.subheader("Predictive Strategy")
st.write("Anticipate when retraining will be needed:")
st.write("- Track performance metrics over time")
st.write("- Monitor feature distribution changes")
st.write("- Schedule retraining before hitting critical thresholds")

# Footer
st.caption("Kobe Bryant Shot Prediction | TDSP Framework Implementation")
