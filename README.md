# Predictive Maintenance & Remaining Useful Life Prediction

A machine learning-powered predictive maintenance system that analyzes industrial machinery sensor data to identify equipment at risk of failure and estimate its remaining operational life. The project combines classification models for failure prediction with regression models for Remaining Useful Life (RUL) estimation, helping organizations reduce unplanned downtime and optimize maintenance schedules.

---

## Features

- Predict equipment failure risk using machine sensor readings
- Estimate Remaining Useful Life (RUL) of machinery
- Automated data preprocessing pipeline
- Feature engineering for operational stress indicators
- Class imbalance handling using SMOTE
- Hyperparameter optimization using GridSearchCV
- Model comparison across multiple algorithms
- SHAP-based explainability for model predictions
- Interactive Streamlit dashboard for visualization and prediction

---

## Dataset

The dataset contains industrial machinery operational records including:

- Temperature measurements
- Vibration levels
- Torque values
- Voltage fluctuations
- Humidity readings
- Maintenance history
- Working hours
- Fault codes
- Machine models
- Failure status labels

---

## Data Preprocessing

The preprocessing pipeline includes:

- Missing value imputation using SimpleImputer
- Outlier detection and treatment using IQR
- Feature importance analysis using Random Forest
- Correlation analysis
- Feature selection
- Custom feature engineering

### Engineered Features

- `Stress_Index`
- `Heat_Plus_Vibration`
- `Heat_Plus_Service`
- `Stress_Hours`

---

## Classification Pipeline

The failure prediction system evaluates:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Training Workflow

- OneHotEncoder for categorical features
- StandardScaler for numerical features
- ColumnTransformer preprocessing pipeline
- SMOTE for class imbalance correction
- Stratified K-Fold Cross Validation
- GridSearchCV hyperparameter tuning

### Evaluation Metrics

- F1 Score
- Recall
- Precision
- Accuracy
- ROC-AUC
- Confusion Matrix

---

## Explainable AI

The project uses SHAP (SHapley Additive exPlanations) to provide interpretable predictions by:

- Identifying features driving failure risk
- Visualizing feature contributions
- Explaining individual machine predictions

---

## Remaining Useful Life Prediction

The RUL component predicts the total expected machine lifetime and calculates:

```text
RUL = Predicted Total Lifetime − Current Working Hours
```

### Regression Models Evaluated

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor

---

## Streamlit Dashboard

The application provides:

- Dataset upload and exploration
- Data preprocessing visualization
- Model training and evaluation
- SHAP explainability dashboards
- Failure risk prediction
- Remaining Useful Life estimation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Seaborn
- Imbalanced-Learn
- Joblib

---

## Business Impact

This solution enables organizations to:

- Reduce unexpected equipment failures
- Minimize production downtime
- Improve maintenance planning
- Extend machinery lifespan
- Support data-driven operational decisions

---

## Author

**Abhishek S**

Machine Learning | Data Science | Predictive Analytics
