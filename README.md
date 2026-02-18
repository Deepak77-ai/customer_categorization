📊 Customer Segmentation & Cluster Prediction System
📌 Project Overview:

- This project demonstrates a real-world customer segmentation system using Clustering + Classification.
- First, customers are grouped into clusters based on similar behavior.
- Then, a machine learning model predicts the most suitable cluster for new customers.
- The final output is a human-readable customer segment, not just a number.
- This approach mimics how businesses perform customer profiling and targeting.



🎯 Problem Statement

Businesses often have customer data but do not know:
- Which customers are high value
- Which customers are regular
- Which customers are low value
- The goal of this project is to:
- Segment customers into meaningful groups using clustering

Predict the correct segment for new customers using a trained model



🧠 Core Idea (Very Important ⭐)

- Clusters are discovered first, then predicted.

- This project does NOT directly classify customers initially.
- Instead, it follows a two-step intelligent approach:

- Unsupervised Learning (Clustering) → Discover patterns

- Supervised Learning (Classification) → Predict patterns for new data



🔁 Project Workflow (Simple & Clear)
1️⃣ Data Understanding & EDA

- Understand features and data distribution

- Detect missing values and outliers

- Identify important behavioral variables



2️⃣ Data Preprocessing

- Handle missing values

- Encode categorical variables (if any)

- Scale numerical features (done properly to avoid data leakage)



3️⃣ Customer Clustering (Unsupervised Learning)

- Applied Agglomerative Clustering

- Customers are grouped based on similarity

- Each customer gets a cluster ID (0, 1, 2, …)

⚠️ Cluster numbers have NO meaning by themselves
They are just identifiers.

Example:

Customer A → Cluster 0
Customer B → Cluster 1
Customer C → Cluster 2



4️⃣ Cluster Interpretation (MOST IMPORTANT STEP ⭐)

Clusters are interpreted using average feature values per cluster.

df.groupby("cluster").mean()


This tells us:

- Which cluster has high income

- Which cluster spends more

- Which cluster purchases frequently

📌 This step converts clusters into business knowledge.



5️⃣ Assigning Business Meaning to Clusters

Based on statistical analysis:

Cluster ID	Customer Segment
0	Low Value Customers
1	Regular Customers
2	High Value Customers

Now clusters become real-world customer segments.



6️⃣ Classification Model (Supervised Learning)

A Logistic Regression model is trained to learn:

Customer Features  →  Cluster ID


This allows the system to:

- Accept new customer data

- Predict which existing cluster they belong to



🤔 What Does “Predicted Cluster” Mean?

When the application shows:

Predicted Cluster: 2

It means:

Based on learned patterns, the model predicts that this new customer belongs to Cluster 2, which represents High Value Customers.

✔ It does NOT create a new cluster
✔ It assigns the customer to an existing segment



🖥️ Application Output (User-Friendly)

Instead of showing only numbers, the app displays:

- Predicted Cluster: 2 → High Value Customers
- This makes the result business-understandable, not technical.



🛠️ Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Streamlit