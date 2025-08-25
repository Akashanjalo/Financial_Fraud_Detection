# 💳 Financial Fraud Detection System (Streamlit App)

This project is an **interactive fraud detection system** built with **Streamlit** and **Python**.  
It combines both **unsupervised learning** (to detect anomalies without labels) and **supervised learning** (to classify fraud when labeled data is available).  
The app helps analyze financial transaction data, identify suspicious activity, and evaluate model performance in a user-friendly way.  

---

## 🚀 Features
- **Upload & Preview Data**: Upload CSV datasets and explore basic statistics.
- **Unsupervised Learning Models**:
  - **Isolation Forest** – detects unusual transactions using tree-based partitioning.
  - **One-Class SVM** – identifies anomalies by learning the boundary of normal data.
  - **Autoencoder (MLP surrogate)** – reconstructs transactions; high reconstruction error = anomaly.
  - Consensus anomaly score combining all three.
- **Supervised Learning Models** (if `is_fraud` column exists):
  - **Logistic Regression**
  - **Random Forest Classifier**
  - Performance metrics: Accuracy, Precision, Recall, F1-score.
  - Confusion Matrix & ROC Curve visualization.
- **Export Results**: Download processed data with anomaly scores and predictions.

---

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)  
- **Machine Learning**: scikit-learn, NumPy, pandas  
- **Visualization**: Matplotlib, Seaborn  

---

## 📂 Installation & Usage

### 1. Clone this repository
```bash
git clone https://github.com/your-username/financial-fraud-detection.git
cd financial-fraud-detection

###  2. Install dependencies
pip install -r requirements.txt

### 3. Run the Streamlit app
streamlit run streamlit_app.py
