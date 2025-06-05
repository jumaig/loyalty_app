
import streamlit as st
import pandas as pd
import joblib
from utils import load_data, preprocess_data, train_models, generate_diagnostics

from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="XGBoost Loyalty Churn Predictor", layout="wide")
st.title("🧠 AI-Powered Loyalty Program Analyzer")

# Upload section
st.sidebar.header("📂 Upload Data")
customer_file = st.sidebar.file_uploader("Upload Customer Info CSV", type="csv")
transaction_file = st.sidebar.file_uploader("Upload Transactions CSV", type="csv")

if customer_file and transaction_file:
    customer_df, transactions_df = load_data(customer_file, transaction_file)

    st.subheader("📊 Raw Data")
    with st.expander("Customer Data"):
        st.dataframe(customer_df.head())
    with st.expander("Transaction Data"):
        st.dataframe(transactions_df.head())

    st.subheader("🔧 Preprocessing & Feature Engineering")
    features_df = preprocess_data(customer_df, transactions_df)
    st.write("Engineered Features:")
    st.dataframe(features_df.head())

    st.subheader("📈 Training XGBoost Model")
    xgb_model, scaler, kmeans, X_test, y_test = train_models(features_df)
    rfm = features_df[['recency', 'monetary', 'points_earned_sum']]
    X_scaled = scaler.transform(rfm)

    generate_diagnostics(xgb_model, X_test, y_test, X_scaled)

    st.subheader("📊 Customer Segmentation")
    features_df['Segment'] = kmeans.predict(X_scaled)
    features_df['Churn_Prediction'] = xgb_model.predict(X_scaled)
    features_df['Churn_Probability'] = xgb_model.predict_proba(X_scaled)[:, 1]

    st.dataframe(features_df[['loyalty_id', 'recency', 'monetary', 'points_earned_sum',
                              'Churn_Prediction', 'Churn_Probability', 'Segment']].head())

    st.subheader("📥 Download Segmentation Results")
    csv = features_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="churn_segmentation_results.csv", mime="text/csv")
else:
    st.info("Please upload both datasets to proceed.")
