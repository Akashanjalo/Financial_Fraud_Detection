# Financial_Fraud_Detection
This project is a Financial Fraud Detection System built with Python and Streamlit that combines both unsupervised learning and supervised learning approaches:

Unsupervised Learning (no labels required):
Used when fraud labels are not available. The app applies:

Isolation Forest → detects unusual points by splitting data into smaller partitions.

One-Class SVM → learns the “normal” region of data and flags points outside it.

Autoencoder (MLP surrogate) → reconstructs transactions; large reconstruction errors indicate anomalies.
These models help in identifying suspicious transactions without needing pre-labeled data.

Supervised Learning (requires fraud labels):
When the dataset contains a column like is_fraud, supervised models are applied:

Logistic Regression

Random Forest Classifier
These models learn directly from past fraud cases and predict fraud on new transactions. The system also shows evaluation metrics like Accuracy, Precision, Recall, F1-score, along with a Confusion Matrix and ROC Curve to measure performance.

✨ Key Features

Upload CSV datasets and preview them interactively.

Detect anomalies using multiple unsupervised models.

Classify fraud using supervised models if labels exist.

Visualize results with heatmaps, ROC curves, and top suspicious records.

Download processed data with anomaly scores and predictions.

🛠 Why this Project is Useful

Fraud detection is a real-world machine learning challenge because:

Fraud data is highly imbalanced (fraud cases are very rare).

Fraudsters constantly change patterns, making it hard for a single model to adapt.

Many datasets are unlabeled, so anomaly detection is often the first step.

By combining unsupervised anomaly detection with supervised classification, this project shows a practical and flexible approach to spotting fraudulent transactions in financial data.
