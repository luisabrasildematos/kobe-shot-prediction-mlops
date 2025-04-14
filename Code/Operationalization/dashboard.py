
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


#define project base path
project_root = Path(__file__).resolve().parent.parent.parent

#define paths
train_path = os.path.join(project_root, "Data", "Processed", "base_train.parquet")
test_path = os.path.join("Data", "Processed", "base_test.parquet")
prod_results_path = os.path.join("project_root", "Data", "Processed", "production_results.parquet")

#load data
@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        return None

train_df = load_data(train_path)
test_df = load_data(test_path)
prod_results = load_data(prod_results_path)

#set up page
st.title("Kobe Bryant Shot Prediction Monitor")
st.markdown('<p style = "font-size:48px;">🏀</p>',
unsafe_allow_html = True)

#add selectedbox for model selection
model_options = ["Logistic Regression", "Decision Tree"]
selected_model = st.selectbox("Select Model", options = model_options)

#performance metrics section
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
    else:
        st.warning("Production results data is missing or doesn't have required columns.")

#model health monitoring
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

#retraining strategies
st.header("3. Retraining Strategies")

#slider for threshold adjustment
threshold = st.slider("Set F1 score threshold", min_value = 0.0,
max_value = 1.0, value = 0.8)

#model training function
def train_model(df, model_type):
    #define features and target
    features = df.drop("shot_made_flag", axis = 1)
    target = df["shot_made_flag"]

    X_train, X_val, y_train, y_val = train_test_split(features, target,
test_size = 0.2, random_state = 42)

    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter = 1000)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    st.write(f"{model_type} Accuracy: {accuracy}")

    return model, X_val, y_val

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

#button for triggering retraining
if st.button("Retrain Model"):
    if train_df is not None:
        with st.spinner('Training model...'):
            model, X_val, y_val = train_model(train_df, selected_model)
        st. success('Model retrained successfully!')
    else:
        st.error("Training data not available")            
#add file uploader
uploaded_file = st.file_uploader("Upload file to predict:")

if uploaded_file is not None:
    try:
        predict_df = pd.read_csv(uploaded_file)
        st.write("Uploaded data:")
        st.dataframe(predict_df.head()) #display first rows

        #make predictions using retrained model if available
        if 'model' in locals():
            predictions = model.predict(predict_df) #use retrained model
            st.write("Predictions:")
            st.write(predictions)
        else:
            st.warning("Please retrain the model first.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

#footer
st.caption("Kobe Bryant Shot Prediction | TDSP Framework Implementation")
