import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def handle_outliers(df,column_name):
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df[column_name])
    plt.title(f"Outliers : {column_name}")
    plt.show()

    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df = df[(df[column_name] >= lower_limit) & (df[column_name] <= upper_limit)]

    return df

def preprocess(signal, df):

    # Dataset info
    print(f"\n{df.info()}")

    # Statistical information of each column
    print(f"\n{df.describe()}")

    # ---------- Handling duplicates ----------
    print(f"\nNumber of duplicates : {df.duplicated().sum()}")
    # No duplicates
    print(f"\nNumber of duplicate machine ID : {df["Machine_ID"].duplicated().sum()}")

    # ---------- Handling inconsistencies ----------
    print(f"\nUnique values of column [Operator_Experience] : {df['Operator_Experience'].unique()}")
    print(f"Unique values of the column [Fault_Code] : {df['Fault_Code'].unique()}")
    # No inconsistencies

    # ---------- Handling missing values ----------
    print(f"\nNumber of missing values :\n{df.isnull().sum()}")
    mode_imputer = SimpleImputer(strategy="most_frequent")
    df[["Fault_Code"]] = mode_imputer.fit_transform(df[["Fault_Code"]])
    print(f"\nNumber of missing values after imputation :\n{df.isnull().sum()}")

    # ---------- Handling outliers ----------
    # df_clean = handle_outliers(df,"Avg_Temperature")
    # df = df_clean.copy()
    # df_clean = handle_outliers(df,"Voltage_Fluctuation")
    # df = df_clean.copy()
    
    # Outliers in [Avg_Temperature] and [Voltage_Fluctuation] are kept as is, because extreme values in these columns could be an indicator of a machine failing
    
    df_clean = handle_outliers(df,"Torque_Nm")
    df = df_clean.copy()

    if (signal == 0) :
        for col in df.select_dtypes(include=["int64","float64"]).columns:
            plt.figure(figsize=(8,5))
            sns.boxplot(x=df[col])
            plt.title(f"Outliers : {col}")
            plt.show()


    # ---------- Skewness ----------
    print(f"\nSkewness in Avg_Temperature : {df["Avg_Temperature"].skew()}")
    print(f"Skewness in Voltage_Fluctuation : {df["Voltage_Fluctuation"].skew()}")
    print(f"Skewness in Vibration_Level : {df["Vibration_Level"].skew()}")
    print(f"Skewness in Torque_Nm : {df["Torque_Nm"].skew()}")
    print(f"Skewness in Working_Hours_Total : {df["Working_Hours_Total"].skew()}")

    # ---------- Handling date format (Feature engineering) ----------
    # Convert to elapsed time for each machine
    df = df.drop("Sensor_Read_Date", axis=1)

    df["Stress_Index"] = df["Vibration_Level"] * df["Torque_Nm"]
    df["Heat_Plus_Vibration"] = df["Avg_Temperature"] * df["Vibration_Level"]
    df["Heat_Plus_Service"] = df["Avg_Temperature"] * df["Last_Service_Days"]
    df["Stress_Hours"]  = df["Stress_Index"] * df["Working_Hours_Total"]

    temp = df.copy()
    temp = temp.drop("Machine_ID", axis=1)

    for col in temp.select_dtypes(include=["object"]).columns:
        temp = pd.get_dummies(temp, columns=[col])
    
    corr = temp.corr()["Failure_Status"].sort_values()
    print(corr)

    # ---------- Feature Importances ----------
    # Rotation_Speed and Oil_Viscosity have near 0 correlation with Failure_Status and Vibration_Level was removed to prevent multicollinearity
    X = df.drop(["Failure_Status","Vibration_Level","Rotating_Speed","Oil_Viscosity","Machine_ID"], axis=1)
    y = df["Failure_Status"]

    # Encode the categorical columns
    X = pd.get_dummies(X,columns=["Machine_Model","Fault_Code","Operator_Experience"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf_model = RandomForestClassifier(random_state=42)

    rf_model.fit(X_train,y_train)

    temp = rf_model.feature_importances_
    x = {"Columns":X.columns,"Importance":temp}
    importance = pd.DataFrame(x).sort_values(by="Importance",ascending=False)
    print("\n",importance)

    important_features = importance[(importance["Importance"] > 0.01)]["Columns"]

    columns = []

    # We are encoding the data later during training as well, so store the original column here.
    # Example : if "Machine_Model" in "Machine_Model_Model_A", append "Machine_Model" to columns[]
    for i in important_features:
        if "Machine_Model" in i and "Machine_Model" not in columns:
            columns.append("Machine_Model")
        elif "Fault_Code" in i and "Fault_Code" not in columns:
            columns.append("Fault_Code")
        elif "Operator_Experience" in i and "Operator_Experience" not in columns:
            columns.append("Operator_Experience")
        else:
            columns.append(i)

    # Include the target column
    if "Failure_Status" not in columns:
        columns.append("Failure_Status")
    
    if "Machine_Model" not in columns:
        columns.append("Machine_Model")

    # Get the cleaned dataset and save it to a CSV file
    df_selected = df[columns].copy()
    print("\n\n\n",df_selected.head())

    df_selected.to_csv("globaltech_machinery_logs_clean.csv")

    if signal == 0:
        # Check for features that are too perfectly aligned with the target
        correlation_matrix = df.select_dtypes(include=[np.number]).corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap: Look for 'Snitch' Features")
        plt.show()

        df_subset = pd.get_dummies(df[["Machine_Model","Operator_Experience","Fault_Code","Failure_Status"]],columns=["Machine_Model","Operator_Experience","Fault_Code"])

        correlation_matrix_2 = df_subset.corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(correlation_matrix_2, annot=True, fmt='.2f', cmap="Blues")
        plt.title("Correlation Heatmap : Find the snitch")
        plt.show()

    if signal == 1:
        return df_selected, importance


if __name__ == "__main__":
    df = pd.read_csv("globaltech_machinery_logs_P1_updated.csv")
    df_clean = preprocess(0,df)