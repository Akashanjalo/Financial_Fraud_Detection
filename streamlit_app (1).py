
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Financial Fraud Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Preprocess numeric columns
    df_clean = df.copy()
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="ignore")
    df_clean = df_clean.dropna(how="all")
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

    st.write("Numeric columns detected:", numeric_cols)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[numeric_cols])

    # ===== Unsupervised Models =====
    st.header("🔍 Unsupervised Fraud Detection")

    # Isolation Forest
    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(X_scaled)
    df_clean['iso_pred'] = iso.predict(X_scaled)
    df_clean['iso_score'] = iso.decision_function(X_scaled)

    # One-Class SVM
    ocsvm = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
    ocsvm.fit(X_scaled)
    df_clean['ocsvm_pred'] = ocsvm.predict(X_scaled)

    # Autoencoder surrogate
    hidden_size = max(1, X_scaled.shape[1] // 2)
    mlp = MLPRegressor(hidden_layer_sizes=(hidden_size,), activation='relu',
                       max_iter=500, random_state=42, early_stopping=True)
    mlp.fit(X_scaled, X_scaled)
    reconstructed = mlp.predict(X_scaled)
    mse = np.mean(np.square(X_scaled - reconstructed), axis=1)
    df_clean['autoencoder_mse'] = mse
    threshold = np.percentile(mse, 99)
    df_clean['autoencoder_flag'] = (df_clean['autoencoder_mse'] > threshold).astype(int)

    # Consensus score
    df_clean['anomaly_score_combined'] = ((df_clean['iso_pred'] == -1).astype(int) +
                                          (df_clean['ocsvm_pred'] == -1).astype(int) +
                                          df_clean['autoencoder_flag'])

    st.subheader("📌 Unsupervised Model Results")
    st.write("IsolationForest anomalies:", (df_clean['iso_pred'] == -1).sum())
    st.write("One-Class SVM anomalies:", (df_clean['ocsvm_pred'] == -1).sum())
    st.write("Autoencoder anomalies:", df_clean['autoencoder_flag'].sum())
    st.write("Top suspicious records:")
    st.dataframe(df_clean.sort_values("anomaly_score_combined", ascending=False).head(10))

    # ===== Supervised Models =====
    if "is_fraud" in df_clean.columns:
        st.header("✅ Supervised Fraud Detection")

        X = df_clean[numeric_cols]
        y = df_clean["is_fraud"]

        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Logistic Regression
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        y_pred_lr = log_reg.predict(X_test)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # Metrics function
        def show_metrics(model_name, y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            st.subheader(f"📊 {model_name} Performance")
            st.write(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
            st.pyplot(fig)

        # Show metrics
        show_metrics("Logistic Regression", y_test, y_pred_lr)
        show_metrics("Random Forest", y_test, y_pred_rf)

        # ROC curve for Random Forest
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
        auc_score = roc_auc_score(y_test, y_prob_rf)
        st.write(f"ROC-AUC Score (Random Forest): {auc_score:.3f}")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    else:
        st.info("⚠️ No 'is_fraud' column found. Supervised models require labeled data.")

    # Download button
    st.download_button(
        label="📥 Download Processed Data",
        data=df_clean.to_csv(index=False).encode("utf-8"),
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )
