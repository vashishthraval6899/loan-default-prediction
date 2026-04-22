# 🚀 Loan Default Prediction using Machine Learning

## 📌 Problem Statement

Defaulted loans can significantly impact the financial health of institutions.
The goal of this project is to **predict the likelihood of a borrower defaulting on a loan** using machine learning techniques.

---

## 🎯 Objectives

* Build a predictive model for loan default
* Handle **missing values, outliers, and imbalanced data**
* Apply **feature engineering**
* Track experiments using **MLflow**
* Implement **Explainability (SHAP)**
* Monitor **Data Drift (PSI, KS, CSI)**
* Deploy using **Flask API + Docker**

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* MLflow
* SHAP
* Flask
* Docker

---

## 🔄 End-to-End Pipeline

Data → Preprocessing → Feature Engineering → Model Training → MLflow → SHAP → Drift Detection → API → Docker

---

## 📊 Exploratory Data Analysis (EDA)

* Checked missing values and distributions
* Identified outliers using boxplots
* Analyzed feature relationships with target variable
* Observed class imbalance in dataset

📌 **EDA Notebook:** `EDA.ipynb`

---

## 🧹 Data Preprocessing

* Missing values handled using **median / mode imputation**
* Outliers handled using **IQR clipping**
* Categorical variables encoded using **OneHotEncoder**
* High-cardinality columns removed to reduce dimensionality

---

## ⚡ Feature Engineering

* Loan to Income Ratio
* Annuity to Income Ratio

These features improved model performance significantly.

---

## ⚖️ Handling Imbalanced Data

* Applied **SMOTE (Synthetic Minority Oversampling Technique)**
* Used **threshold tuning** to improve recall

---

## 🤖 Models Used

| Model                    | Purpose               |
| ------------------------ | --------------------- |
| Logistic Regression      | Baseline              |
| Logistic Regression (L1) | Feature selection     |
| XGBoost                  | Final optimized model |

---

## 📈 Evaluation Metrics

* ROC-AUC
* Precision
* Recall
* F1 Score

📌 **Key Insight:**

* XGBoost achieved highest AUC (~0.7+)
* Threshold tuning improved recall significantly
* Logistic model provided interpretability

---

## 🔬 MLflow Experiment Tracking

* Logged parameters, metrics, and models
* Compared multiple experiments
* Stored SHAP plots and drift metrics

📸 *(Add MLflow screenshots here)*

---

## 🧠 Explainability (SHAP)

* Used SHAP to interpret feature importance
* Identified key factors influencing loan default

📸 *(Add SHAP plot screenshot here)*

---

## 📉 Data Drift Monitoring

* PSI (Population Stability Index)
* KS (Kolmogorov-Smirnov)
* CSI (Characteristic Stability Index)

📌 Result: Data distribution is stable → model is reliable

---

## 🌐 API Deployment (Flask)

### Endpoint:

```bash
POST /predict
```

### Example Request:

```json
{
  "Client_Income": 50000,
  "Credit_Amount": 200000,
  "Loan_Annuity": 10000
}
```

### Example Response:

```json
{
  "prediction": 0,
  "default_probability": 0.23
}
```

---

## 🐳 Docker Deployment

```bash
docker build -t loan-app .
docker run -p 5000:5000 loan-app
```

---

## ▶️ How to Run Project

```bash
pip install -r requirements.txt
python model.py
mlflow ui
python app.py
```

---

## 📁 Project Structure

```
project/
│
├── app.py
├── model.py
├── drift.py
├── explain.py
├── requirements.txt
├── Dockerfile
├── EDA.ipynb
├── README.md
├── data/
└── mlruns/
```

---

## 🚀 Future Improvements

* Hyperparameter tuning using Optuna
* Streamlit dashboard for visualization
* Real-time model monitoring
* CI/CD pipeline

---

## 🏆 Conclusion

This project demonstrates a **complete end-to-end machine learning pipeline**, including:

✔ Data preprocessing
✔ Feature engineering
✔ Model building
✔ Explainability (SHAP)
✔ Drift monitoring
✔ Deployment

---

## 👨‍💻 Author

**Vashishth Raval**
