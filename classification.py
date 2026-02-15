import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score, roc_curve, precision_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import shap

def train_classification(signal,df):

    # ----------- Remove the first useless column -----------
    if df.columns[0].startswith("Unnamed: 0"):
        df.drop(df.columns[0], axis=1, inplace=True)

    '''
    df["Temp_Change"] = df.groupby("Machine_Model")["Avg_Temperature"].diff().fillna(0).reset_index(0, drop=True)
    df["Torque_Change"] = df.groupby("Machine_Model")["Torque_Nm"].diff().fillna(0).reset_index(0, drop=True)
    df["Voltage_Change"] = df.groupby("Machine_Model")["Voltage_Fluctuation"].diff().fillna(0).reset_index(0, drop=True)

    df["Cum_Torque"]  = df.groupby('Machine_Model')['Torque_Nm'].cumsum().reset_index(0, drop=True)
    df["Cum_Temp"] = df.groupby('Machine_Model')['Avg_Temperature'].cumsum().reset_index(0, drop=True)
    df["Cum_Stress_Index"] = df.groupby("Machine_Model")["Stress_Index"].cumsum().reset_index(0, drop=True)
    df["Cum_Voltage_Fluc"] = df.groupby("Machine_Model")["Voltage_Fluctuation"].cumsum().reset_index(0, drop=True)

    df["Torque_RollMean_10"] = df.groupby('Machine_Model')['Torque_Nm'].rolling(10).mean().fillna(0).reset_index(0, drop=True)

    df["Avg_Temperature_Roll"] = df.groupby("Machine_Model")["Avg_Temperature"].rolling(10).max().fillna(0).reset_index(0, drop=True)

    df["Voltage_Roll"] = df.groupby("Machine_Model")["Voltage_Fluctuation"].rolling(10).std().fillna(0).reset_index(0, drop=True)

    df["Stress_Roll"] = df.groupby("Machine_Model")["Stress_Index"].rolling(20).mean().fillna(0).reset_index(0, drop=True)

    df["Temp_Change_Roll"] = df.groupby("Machine_Model")["Temp_Change"].rolling(10).max().fillna(0).reset_index(0, drop=True)

    df["Voltage_Change_Roll"] = df.groupby("Machine_Model")["Voltage_Change"].rolling(10).mean().fillna(0).reset_index(0, drop=True)

    df["Temp_x_Torque_Cum"] = df["Avg_Temperature"] * df["Cum_Torque"]
    df["Stress_x_Voltage_Roll"] = df["Stress_Index"] * df["Voltage_Roll"]
    df["Temp_Change_x_Torque_Change"] = df["Temp_Change"] * df["Torque_Change"]

    df["Temp_x_Torque"] = df["Avg_Temperature"] * df["Torque_Nm"]
    '''

    #window = 2
    
    #df["Rolling_Temp"] = df.groupby("Machine_Model")["Avg_Temperature"].rolling(window).mean().fillna(0).reset_index(0, drop=True)
    #df["Rolling_Voltage"] = df.groupby("Machine_Model")["Voltage_Fluctuation"].rolling(window).mean().fillna(0).reset_index(0, drop=True)

    #print(f"\n{df[["Rolling_Temp","Failure_Status"]].corr()}")
    #print(f"\n{df[["Rolling_Voltage","Failure_Status"]].corr()}")

    df = df.drop(["Machine_Model","Stress_Hours"], axis=1)
    
    # ----------- Split the data -----------
    # Drop the target column (Failure_Status) and the columns that would potentially leak data to the model
    X = df.drop(["Failure_Status"], axis=1)
    y = df["Failure_Status"]

    print(X.columns)

    #Index(['Heat_Plus_Vibration', 'Last_Service_Days', 'Stress_Index',
       #'#Avg_Temperature', 'Temp_Change', 'Torque_Nm', 'Working_Hours_Total',
       #'Ambient_Humidity', 'Voltage_Fluctuation', 'Time_Elapsed_Days',
       #'Machine_Model'],
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("\n\n",y_train.value_counts(normalize=True)*100)

    cat_columns = X.select_dtypes(include=["object"]).columns
    num_columns = X.select_dtypes(include=["int64","float64"]).columns

    # Use ColumnTransformer to scale numeric features and encode categorical ones

    # handle_unknown="ignore" -> During prediction, if the user enters a value the model isn't familiar with, it simply ignores it
    preprocess_data = ColumnTransformer(transformers=[
        ("one_hot_encode", OneHotEncoder(handle_unknown="ignore"), cat_columns),
        ("standard_scaler", StandardScaler(), num_columns)
    ])

    # These are the three models to test. The name of the model is the key. The values are a list, that contain the model declaration/initialization at the 0th index and the various hyperparameters to test in the 1st index
    models = {
        "Logistic Regression" : [LogisticRegression(class_weight="balanced",random_state=42),
                                 {"classifier__C":[0.1, 0.3, 0.7, 1, 10]}],

        "Random Forest Classifier" : [RandomForestClassifier(random_state=42),
                                      {"classifier__max_depth":[20,50,100]}],


        "XGBoost Classifier" : [XGBClassifier(eval_metric="logloss",scale_pos_weight=20,random_state=42),{"classifier__n_estimators":[20,50,100,200],"classifier__learning_rate" : [0.1,0.3,0.5,0.7,1],"classifier__max_depth": [3, 5, 7],
        "classifier__subsample": [0.8, 1],
        "classifier__colsample_bytree": [0.8, 1]}],
    }

    f1_score_final = 0
    final_model = None

    for model_train in models:

        # The pipeline that automates the process. First it will preprocess the data, then apply SMOTE to handle imbalance, and then initialize the model
        pipeline = ImbPipeline([
            ("preprocess",preprocess_data),
            ("smote",SMOTE(random_state=42)),
            ("classifier",models[model_train][0])
        ])

        # Its going to see that 'failures' make up 5% of the data, so when it splits the dataset into 5 random folds, it ensures this same 5% amount of ''failures' is present in each fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # n_jobs=-1 -> indicates use all the CPU cores for faster execution
        gcv = GridSearchCV(
            estimator=pipeline,
            param_grid=models[model_train][1],
            cv=skf,
            scoring={"f1":"f1","accuracy":"accuracy","recall":"recall"},
            refit="f1",
            n_jobs=-1
        )

        gcv.fit(X_train, y_train)

        print(f"\n{model_train} : Best F1 : {gcv.cv_results_['mean_test_f1'][gcv.best_index_]}")
        print(f"\n{model_train} : Best Accuracy : {gcv.cv_results_['mean_test_accuracy'][gcv.best_index_]}")
        print(f"\n{model_train} : Best Recall : {gcv.cv_results_['mean_test_recall'][gcv.best_index_]}")

        if (gcv.best_score_ > f1_score_final):
            f1_score_final = gcv.best_score_
            final_model = gcv.best_estimator_


    print(f"\nBest model params :\n{gcv.best_params_}")
    print("-" * 30)
    print(f"WINNING MODEL: {final_model.named_steps['classifier']}")
    print("-" * 30)

    model_for_shap = final_model.named_steps['classifier']
    X_test_transformed = final_model.named_steps['preprocess'].transform(X_test)
    X_subset = X_test_transformed[:500]
    num_features = num_columns.to_list()  # from your earlier code
    all_features = num_features
    explainer = shap.Explainer(model_for_shap, X_test_transformed, feature_names=all_features)
    shap_values = explainer(X_subset)
    # Extract the classifier only
    
    # Waterfall for first 3 rows
    num_machines = 3
    for i in range(num_machines):
        print(f"\nWaterfall Explanation for Customer #{i}")
        shap.plots.waterfall(shap_values[i])

    y_probabilities = final_model.predict_proba(X_test)
    y_pred = final_model.predict(X_test)
    failures = y_probabilities[:, 1]

    roc_auc = roc_auc_score(y_test, failures)
    print(f"\nROC-AUC Score : {roc_auc}")

    fpr, tpr, thresholds = roc_curve(y_test, failures)

    if (signal == 0):
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
        plt.plot([0,1], [0,1], linestyle="--")  # random baseline
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    print(failures)
    con_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"\nF1 Score : {f1_score(y_test, y_pred)}")
    print("\n",report)
    print(f"\nAccuracy : {accuracy_score(y_test, y_pred)}")

    plt.figure(figsize=(5,5))
    sns.heatmap(con_matrix, annot=True, fmt='d', cmap="Greens")
    plt.title("Confusion Matrix")
    plt.show()

    file = "classification_model.pkl"
    joblib.dump(final_model,file)

    if signal == 1:
        return (plt,report)


def predict(df,input_list):
    model = joblib.load("classification_model.pkl")

    df_predict = df.drop(["Failure_Status","Machine_Model","Stress_Hours"], axis=1)
    x = pd.DataFrame([input_list],columns=df_predict.columns)

    failed = model.predict(x)
    predict_probability = model.predict_proba(x)[:,1]
    return (failed,predict_probability)


if __name__ == "__main__":
    df_clean = pd.read_csv("globaltech_machinery_logs_clean.csv")
    train_classification(0,df_clean)