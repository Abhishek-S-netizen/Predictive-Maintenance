import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib


def regression(df):

    final_rmse = -np.inf
    final_model = None

    if df.columns[0].startswith("Unnamed: 0"):
        df = df.drop(df.columns[0], axis=1)
    
    df = df.drop(["Machine_Model","Stress_Hours"], axis=1)

    df_failed = df[df["Failure_Status"] == 1]

    # .idmax gives the index of the row with maximum Working_Hours_Total
    # .loc basically chooses all the columns
    # .reset_index() resets the index 
    # At different points of time each machine will have a "Working_Hours_Total". We need to get the maximum value for each machine, because that value is basically the total number of hours the machine has worked overall, right before it failed

    print(df_failed["Working_Hours_Total"].describe())
    scaler_temp = StandardScaler()

    # Noting correlations
    temp = df_failed.copy()
    for col in temp.select_dtypes(include=["object"]).columns:
        temp = pd.get_dummies(temp, columns=[col])
    
    correlation_matrix = temp.corr()
    wht_corr = temp.corr()["Working_Hours_Total"].sort_values()
    print(wht_corr)
    print(correlation_matrix)

    plt.figure(figsize=(10,5))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="Greens")
    plt.title("Correlation")
    plt.show()

    # Splitting the data
    X_train = df_failed.drop(["Failure_Status","Working_Hours_Total"], axis=1)
    y_train = df_failed["Working_Hours_Total"]
    num_columns = X_train.select_dtypes(include=["int64","float64"]).columns
    cat_columns = X_train.select_dtypes(include=["object"]).columns

    print(X_train.columns)

    # X = df_failed.drop(["Failure_Status","Working_Hours_Total"], axis=1)
    # y = df_failed["Working_Hours_Total"]
    # num_columns = X.select_dtypes(include=["int64","float64"]).columns
    # cat_columns = X.select_dtypes(include=["object"]).columns
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Use ColumnTransfomer to encode categorical columns and scale numeric columns
    preprocess_data = ColumnTransformer(transformers=[
        ("encoder",OneHotEncoder(handle_unknown="ignore"),cat_columns),
        ("scaler",StandardScaler(),num_columns)
    ])

    models = {
        "Linear Regression":[LinearRegression(),{}],

        "Ridge Regression":[Ridge(random_state=42),{"regressor__alpha":[0.1,0.3,0.5,0.7,1.0,3.0,5.0,10.0]}],

        "Lasso Regression":[Lasso(random_state=42),{"regressor__alpha":[0.1,0.3,0.5,0.7,1.0,3.0,5.0,10.0]}],

        "Random Forest Regressor":[RandomForestRegressor(random_state=42),{"regressor__max_depth":[20,50,100,200], "regressor__n_estimators":[20,50,100,200]}],

        "XGBoost Regressor":[XGBRegressor(eval_metric="rmse",random_state=42),{"regressor__n_estimators":[20,50,100,200],"regressor__learning_rate":[0.1,0.2,0.5,0.7,1]}]
    }


    for model_train in models:
        pipeline = Pipeline([
            ("preprocess",preprocess_data),
            ("regressor",models[model_train][0])
        ])

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        gcv = GridSearchCV(
            estimator=pipeline,
            param_grid=models[model_train][1],
            cv=kf,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        gcv.fit(X_train, y_train)

        print(f"\n{model_train} : Best RMSE : {gcv.best_score_}")

        if gcv.best_score_ > final_rmse:
            final_rmse = gcv.best_score_
            final_model = gcv.best_estimator_


    print("*" * 30)
    print(F"WINNER : {final_model.named_steps['regressor']}")
    print("*" * 30)
    
    df_alive = df[df["Failure_Status"] == 0]
    X_test = df_alive.drop(["Failure_Status","Working_Hours_Total"], axis=1)

    y_pred = final_model.predict(X_test)

    print(f"\n\n{y_pred - df_alive["Working_Hours_Total"]}")

    file = "regression_model.pkl"
    joblib.dump(final_model,file)

    #print(f"\nMAE {mean_absolute_error(y_test, y_pred)}")
    #print(f"\nMSE : {mean_squared_error(y_test, y_pred)}")
    #print(f"\nR Score : {r2_score(y_test, y_pred)}")


def predict(df,input_list,current_working_hours):
    model = joblib.load("regression_model.pkl")

    df_predict = df.drop(["Failure_Status","Machine_Model","Stress_Hours","Working_Hours_Total"], axis=1)
    x = pd.DataFrame([input_list],columns=df_predict.columns)

    rul_predict = model.predict(x)[0]
    rul = rul_predict - current_working_hours
    return rul


if __name__ == "__main__":
    df = pd.read_csv("globaltech_machinery_logs_clean.csv")
    regression(df)