import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Page Title
st.set_page_config(page_title="📊 Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("An interactive machine learning application for customer churn analysis!")

# File Upload
uploaded_file = st.file_uploader("📂 Upload Your CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📌 Dataset Preview:", df.head())

    # Handle missing values
    df.dropna(inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Define Features & Target
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Sidebar Options
    model_choice = st.sidebar.radio("🔹 Choose Model", ("Random Forest", "XGBoost"))
    st.sidebar.write("🎯 Selected Model:", model_choice)

    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    else:
        model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, preds)
    st.sidebar.write(f"📊 Model Accuracy: **{acc:.4f}**")

    # Classification Report
    st.subheader("📄 Classification Report")
    st.text(classification_report(y_test, preds))

    # Feature Importance
    feature_importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

    # Plot Feature Importance
    st.subheader("📈 Feature Importance Visualization")
    fig = px.bar(feature_importance_df, x="Feature", y="Importance", color="Importance",
                 color_continuous_scale="viridis", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

    # User Input for Prediction
    st.subheader("📌 Predict on New Data")
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.number_input(f"Enter {feature}", value=float(X_test[0][list(X.columns).index(feature)]))

    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([input_data])
        input_df_scaled = scaler.transform(input_df)
        prediction = model.predict(input_df_scaled)
        churn_status = "❌ Churn" if prediction[0] == 1 else "✅ No Churn"
        st.success(f"📝 Prediction: **{churn_status}**")

# Footer
st.markdown("📌 Developed by **Md Anique Zzama** 🚀")
