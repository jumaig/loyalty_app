
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def load_data(customer_file, transaction_file):
    customer_df = pd.read_csv(customer_file, encoding='ISO-8859-1')
    transactions_df = pd.read_csv(transaction_file, encoding='ISO-8859-1')
    return customer_df, transactions_df

def preprocess_data(customer_df, transactions_df):
    customer_df.columns = customer_df.columns.str.lower().str.strip()
    transactions_df.columns = transactions_df.columns.str.lower().str.strip()
    customer_df['registered_date'] = pd.to_datetime(customer_df['registered_date'], errors='coerce')
    transactions_df['ticket_date'] = pd.to_datetime(transactions_df['ticket_date'], errors='coerce')
    customer_df['loyalty_id'] = customer_df['loyalty_id'].astype(str)
    transactions_df['loyalty_id'] = transactions_df['loyalty_id'].astype(str)

    numeric_cols = ['qty_sold_unit', 'revenue_value', 'unit_price', 'discount', 'points_earned']
    for col in numeric_cols:
        transactions_df[col] = (
            transactions_df[col].astype(str)
            .str.replace(',', '', regex=False)
            .str.strip()
        )
        transactions_df[col] = pd.to_numeric(transactions_df[col], errors='coerce')

    transactions_df = transactions_df[transactions_df['qty_sold_unit'] > 0]
    transactions_df = transactions_df[transactions_df['revenue_value'] > 0]
    transactions_df.fillna({'discount': 0, 'points_earned': 0}, inplace=True)
    transactions_df['unit_price'].fillna(transactions_df['unit_price'].median(), inplace=True)

    merged = transactions_df.merge(customer_df[['loyalty_id']], on='loyalty_id', how='inner')

    features = merged.groupby('loyalty_id').agg({
        'revenue_value': ['sum', 'mean'],
        'qty_sold_unit': 'sum',
        'points_earned': 'sum',
        'ticket_date': ['min', 'max', 'count']
    })
    features.columns = ['_'.join(col) for col in features.columns]
    features.reset_index(inplace=True)
    features['recency'] = (merged['ticket_date'].max() - features['ticket_date_max']).dt.days
    features['frequency'] = features['ticket_date_count']
    features['monetary'] = features['revenue_value_sum']
    features['loyalCustomer'] = (features['frequency'] > features['frequency'].median()).astype(int)

    return features

def train_models(features_df):
    rfm = features_df[['recency', 'monetary', 'points_earned_sum']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm)
    y = features_df['loyalCustomer']

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, 'xgboost_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)
    joblib.dump(kmeans, 'kmeans_model.joblib')

    return xgb_model, scaler, kmeans, X_test, y_test

def generate_diagnostics(xgb_model, X_test, y_test, X_scaled):
    st.subheader("Model Evaluation")

    st.write("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, xgb_model.predict(X_test)), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.write("ROC Curve")
    RocCurveDisplay.from_estimator(xgb_model, X_test, y_test)
    st.pyplot(plt.gcf())

    st.write("KMeans Elbow & Silhouette Scores")
    inertia = []
    silhouette = []
    K = range(2, 10)
    for k in K:
        model = KMeans(n_clusters=k, random_state=42)
        preds = model.fit_predict(X_scaled)
        inertia.append(model.inertia_)
        silhouette.append(silhouette_score(X_scaled, preds))

    fig2, ax2 = plt.subplots()
    ax2.plot(K, inertia, marker='o')
    ax2.set_title("Elbow Method")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Inertia")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.plot(K, silhouette, marker='s', color='green')
    ax3.set_title("Silhouette Scores")
    ax3.set_xlabel("k")
    ax3.set_ylabel("Score")
    st.pyplot(fig3)
