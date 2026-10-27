RetentionIQ is an end-to-end machine learning project designed to predict customer churn and help businesses identify customers who may be at risk of leaving.

The project combines data analysis, machine learning, interactive visualizations, and model interpretation in a Streamlit dashboard.

Features

Data Analysis: Explore customer demographics, services, contracts, and billing information.

Churn Prediction: Estimate the probability that a customer will leave.

Customer Segmentation: Analyze churn patterns across customer groups.

Model Evaluation: Evaluate performance using ROC-AUC, PR-AUC, precision, recall, and F1-score.

Model Interpretation: Use permutation importance to understand which features contribute most to predictive performance.

Interactive Dashboard: Explore the data and make predictions through a Streamlit web app.

Machine Learning

The project compares Logistic Regression and Random Forest models using a consistent training, validation, and test split.

The selected model is saved as a reusable scikit-learn pipeline that includes preprocessing and prediction.

Technology Stack

Python

Pandas and NumPy

Scikit-learn

Plotly

Streamlit

Jupyter Notebook

Joblib

Project Structure

RetentionIQ/
├── App/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
│       └── telco_clean.csv
├── models/
│   └── best_model.joblib
├── notebooks/
├── requirements.txt
├── README.md
└── .gitignore

Dataset

This project uses the IBM Telco Customer Churn dataset, which contains customer account, service, contract, and billing information.

The target variable is Churn, which indicates whether a customer left the company.

Run Locally

Clone the repository.

Install the required dependencies:

pip install -r requirements.txt

Start the Streamlit application:

streamlit run App/app.py

Evaluation

The final model is evaluated on a held-out test set to estimate its performance on unseen customers. Evaluation includes ROC-AUC, PR-AUC, precision, recall, and F1-score.

Disclaimer

Predictions are estimates based on historical data and should support—not replace—business judgment. Feature importance indicates predictive usefulness, not causation.

Author

Developed as a data science and machine learning portfolio project.
