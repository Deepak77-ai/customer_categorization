# 📊 Customer Segmentation & Prediction App

> A machine learning web app that predicts customer segments based on demographics and purchase behavior — built with K-Means clustering + Logistic Regression, deployed on Streamlit.

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://customercategorization-r6fh22kyxtv7zyctvtteay.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

---

## 🚀 Live Demo

👉 **[Open the App](https://customercategorization-r6fh22kyxtv7zyctvtteay.streamlit.app/)** *(may take a few seconds to load)*

---

## 📸 Screenshots

| Input Form |  |
|:---:|:---:|
| <img width="1909" height="916" alt="Screenshot 2026-05-30 143022" src="https://github.com/user-attachments/assets/79d810cf-c68a-4abf-aac4-d613e4723e5e" />

| Prediction Result |
 | <img width="1909" height="913" alt="Screenshot 2026-05-30 143116" src="https://github.com/user-attachments/assets/e4c6d46e-0e2e-4d62-9938-e82cada7073e" />
 |

> Fill in customer details (age, income, spending habits) → click **Predict Customer Cluster** → instantly see the segment, recommended offers, and business action.

---

## 🧠 What It Does

This app takes customer details as input and classifies them into one of **3 segments**:

| Cluster | Type | Description |
|:---:|---|---|
| 🔴 0 | **Low Value Customer** | Price-sensitive, infrequent buyer, low spending |
| 🟡 1 | **Medium Value Customer** | Regular buyer, moderate spending, stable engagement |
| 🟢 2 | **High Value Customer** | Loyal, frequent, high-spending premium customer |

Each prediction also returns:
- ✅ Customer type label
- 🎁 Recommended offers (loyalty points, seasonal deals, etc.)
- 📌 Business recommendation (e.g., "Encourage repeat purchases using loyalty programs")

---

## 🏗️ How It Works

```
Raw Customer Data
       ↓
  K-Means Clustering  ←── Unsupervised segmentation
       ↓
  Logistic Regression ←── Supervised classification (85–90% accuracy)
       ↓
  Streamlit Web App   ←── User-facing prediction interface
```

**Model Pipeline:**
- **K-Means** groups existing customers into 3 clusters based on spending and behavior
- **Logistic Regression** (with `GridSearchCV` hyperparameter tuning) learns to classify new customers into those clusters
- Cross-validated accuracy: **~85–90%**

---

## 📥 Input Features

| Category | Features |
|---|---|
| **Demographics** | Age, Education Level, Marital Status, Parental Status, No. of Children, Annual Income |
| **Purchase Behavior** | Total Spending, Days as Customer, Recency (days since last purchase) |
| **Product Spending** | Wines, Fruits, Meat, Fish, Sweets, Gold |
| **Channels** | Web, Catalog, Store purchases; Web visits/month |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| scikit-learn | K-Means, Logistic Regression, GridSearchCV |
| Streamlit | Web app UI |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | EDA & visualization |

---

## ⚙️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Deepak77-ai/customer_categorization.git
cd customer_categorization
```

**2. Create & activate a conda environment**
```bash
conda create --prefix venv python -y
conda activate venv/
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Launch the app**
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
customer_categorization/
├── app.py                  # Streamlit application
├── customer_model.pkl      # Trained Logistic Regression model
├── clustered_data.csv      # Dataset with cluster labels
├── marketing_campaign.csv  # Raw dataset
├── main_part1.ipynb        # EDA & preprocessing
├── main_part2.ipynb        # K-Means clustering
├── main_part3.ipynb        # Classification model training
└── requirements.txt
```

---

## 📊 Dataset

[marketing_campaign.csv](https://github.com/entbappy/Branching-tutorial/blob/master/marketing_campaign.zip) — Contains customer demographics, product spending, and channel usage across ~2,200 records.

---

## 📬 Contact

Made by **Deepak** · [GitHub](https://github.com/Deepak77-ai)
