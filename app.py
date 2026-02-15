import pandas as pd
import numpy as np
import streamlit as st
import io
import classification
import regression
import preprocess_data
from PIL import Image

st.set_page_config(layout="wide")
st.title("Predictive Maintenance - Global-Tech Industrial Solutions")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Upload dataset","Preprocessed data","Preprocessing", "Training the models", "Logistic Regression","Evaluation", "RUL Prediction", "Predict"])

if "cleaned_dataset" not in st.session_state:
    st.session_state.cleaned_dataset = None


with tab1:
    file = st.file_uploader("Upload CSV file",type='csv')

    if file is None:
        file = "globaltech_machinery_logs_P1_updated.csv"
        df = pd.read_csv(file)
        st.info("Using default dataset")
    else:
        df = pd.read_csv(file)
    
    st.dataframe(df)    
    st.session_state.cleaned_dataset, importance = preprocess_data.preprocess(1,df)
    st.markdown("<br><br>",unsafe_allow_html=True)
    st.subheader("Feature Importance")
    st.dataframe(importance)

with tab2:
    if st.session_state.cleaned_dataset is not None:
        st.dataframe(st.session_state.cleaned_dataset)

        buffer = io.StringIO()
        st.session_state.cleaned_dataset.info(buf=buffer)
        info_value = buffer.getvalue()

        st.markdown("<br><br>",unsafe_allow_html=True)

        st.subheader("Summary of data")
        st.code(info_value)

        st.markdown("<br><br>",unsafe_allow_html=True)

        st.subheader("Summary statistics of numerical columns")
        st.write(st.session_state.cleaned_dataset.describe())

with tab3:
    if st.session_state.cleaned_dataset is not None:
        st.header("Process")
        st.subheader("Handling duplicates")
        st.text("There were no duplicates found in the dataset")

        st.markdown("<br><br>",unsafe_allow_html=True)
        st.subheader("Handling inconsistencies")
        st.text("There were no inconsistencies found in the dataset")

        st.markdown("<br><br>",unsafe_allow_html=True)
        st.subheader("Handling missing values")
        st.text("Missing values where found in the column [Fault_Code] and they were handled using SimpleImputer with strategy='most_frequent'")
        
        st.code('mode_imputer = SimpleImputer(strategy="most_frequent")\
    \ndf[["Fault_Code"]] = mode_imputer.fit_transform(df[["Fault_Code"]])')

        st.markdown("<br><br>",unsafe_allow_html=True)
        st.subheader("Handling outliers")
        st.text("Outliers were found in the columns [Avg_Temperature], [Voltage_Fluctuation], [Torque_Nm]. Outliers of the first two columns were kept as is, since in their case, extreme values could be a an indicator of failure. Outliers for [Torque_Nm] were handled using the IQR method")

        st.code("def handle_outliers(df,column_name):\
                \n\tplt.figure(figsize=(8,5))\
                \n\tsns.boxplot(x=df[column_name])\
                \n\tplt.title(f'Outliers : {column_name}')\
                \n\tplt.show()\
                \n\n\tQ1 = df[column_name].quantile(0.25)\
                \n\tQ3 = df[column_name].quantile(0.75)\
                \n\n\tIQR = Q3 - Q1\
                \n\tlower_limit = Q1 - 1.5 * IQR\
                \n\tupper_limit = Q3 + 1.5 * IQR\
                \n\n\tdf = df[(df[column_name] >= lower_limit) & (df[column_name] <= upper_limit)]\
                \n\n\treturn df\
                \n\ndf_clean = handle_outliers(df,'Torque_Nm')\
                \ndf = df_clean.copy()")
        
        st.markdown("<br><br>",unsafe_allow_html=True)
        st.subheader("Feature Engineering")
        st.text("Interaction features were created from existing sensor features")
        st.code("df['Stress_Index'] = df['Vibration_Level'] * df['Torque_Nm']\
                \ndf['Heat_Plus_Vibration'] = df['Avg_Temperature'] * df['Vibration_Level']\
                \ndf['Heat_Plus_Service'] = df['Avg_Temperature'] * df['Last_Service_Days']\
                \ndf['Stress_Hours']  = df['Stress_Index'] * df['Working_Hours_Total']")
        

with tab4:
    if st.session_state.cleaned_dataset is not None:
        
        st.subheader("Training models for classification")
        st.text("Models trained included Logistic Regression, RandomForestClassifier, and XGBoost Classifier. Target columns was Failure_Status.")
        st.markdown("""
**Data preprocessing**
- Categorical columns were encoded using OneHotEncoder
- Numeric columns were scaled using StandardScaler
- ColumnTransformer was used to apply encoding and scaling
- SMOTE was used to handle imbalance in the data

""")
        
        st.code("preprocess_data = ColumnTransformer(transformers=[\
                \n\t('one_hot_encode', OneHotEncoder(handle_unknown='ignore'), cat_columns),\
                \n\t('standard_scaler', StandardScaler(), num_columns)\
                \n])")

        st.markdown("<br>",unsafe_allow_html=True)
        
        st.markdown("""
**Model training**
- Pipeline was used to combine preprocessing and SMOTE and the model was created
- Cross-validation was done using GridSearchCV and StratifiedKFold
- The model with the best F1 Score (the balance between recall and precision) is chosen as the best one

""")
        st.code(" models = {\
                \n\t'Logistic Regression' : [LogisticRegression(class_weight='balanced',random_state=42),{'classifier__C':[0.1, 0.3, 0.7, 1, 10]}],\
                \n\t'Random Forest Classifier' : [RandomForestClassifier(random_state=42), {'classifier__max_depth':[20,50,100]}],\
                \n\t'XGBoost Classifier' : [XGBClassifier(eval_metric='logloss',scale_pos_weight=20 random_state=42),{'classifier__n_estimators':[20,50,100,200]\
                \n\t\t\t'classifier__learning_rate' : [0.1,0.3,0.5,0.7,1],'classifier__max_depth': [3, 5, 7],\
                \n\t\t\t'classifier__subsample': [0.8, 1],'classifier__colsample_bytree': [0.8, 1]}]}\
                \n\nf1_score_final = 0\
                \nfinal_model = None\
                \n\nfor model_train in models:\
                    \n\t#The pipeline that automates the process. First it will preprocess the data, then apply SMOTE to handle imbalance, and then initialize the model\
                    \n\tpipeline = ImbPipeline([\
                        \n\t\t('preprocess',preprocess_data),\
                        \n\t\t('smote',SMOTE(random_state=42)),\
                        \n\t\t('classifier',models[model_train][0])\
                    \n\t])\
                    \n\n\t# Its going to see that 'failures' make up 5% of the data, so when it splits the dataset into 5 random folds,\n\t# it ensures this same 5% amount of 'failures' is present in each fold\
                    \n\tskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\
                    \n\t# n_jobs=-1 -> indicates use all the CPU cores for faster execution\
                    \n\tgcv = GridSearchCV(\
                        \n\t\testimator=pipeline,\
                        \n\t\tparam_grid=models[model_train][1],\
                        \n\t\tcv=skf,\
                        \n\t\tscoring={'f1':'f1','accuracy':'accuracy','recall':'recall'},\
                        \n\t\trefit='f1',\
                        \n\t\tn_jobs=-1\
                    )\
                    \n\n\tgcv.fit(X_train, y_train)\
                    \n\tprint(f'{model_train} : Best F1 : {gcv.cv_results_['mean_test_f1'][gcv.best_index_]}')\
                    \n\tprint(f'{model_train} : Best Accuracy : {gcv.cv_results_['mean_test_accuracy'][gcv.best_index_]}')\
                    \n\tprint(f'{model_train} : Best Recall : {gcv.cv_results_['mean_test_recall'][gcv.best_index_]}')\
                    \n\n\tif (gcv.best_score_ > f1_score_final):\
                        \n\t\tf1_score_final = gcv.best_score_\
                        \n\t\tfinal_model = gcv.best_estimator_")
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("Training models for regression")
        st.text("For predicting the Remaining Useful Life, the same process was followed. Preprocess with ColumnTransformer, cross-validate with GridSearchCV and KFold, and get the best model based on RMSE (Root-Mean-Squared-Error). Models trained were Linear Regression, Ridge, Lasso, RandomForestRegressor and XGBoost Regressor and the target column was Working_Hours_Total.")

with tab5:
    if st.session_state.cleaned_dataset is not None:
        st.header("Train models")
        clicked = st.button("Train models (Logistic Regression, RandomForestClasifier, XGBoostClassifier)")
        st.markdown("<br><br>",unsafe_allow_html=True)

        if clicked:
            plot, report = classification.train_classification(1,st.session_state.cleaned_dataset)
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot)
            with col2:
                st.code(report)
        else:
            st.code("Logistic Regression was the model chosen for proceeding with predictions")
            st.markdown("<br><br>",unsafe_allow_html=True)
            col1, col2 = st.columns(2)            
            with col1:
                image = Image.open("Pictures/Confusion_Matrix.png")
                st.image(image, width=550)
            with col2:
                st.code("Recall (1) : 0.73\
                        \nRecall (0) : 0.72\
                        \nPrecision (1) : 0.40\
                        \nPrecision (0) : 0.91\
                        \nF1-Score (1) : 0.51\
                        \nF1-Score (0) : 0.80")

                st.text("VERSUS")
                st.code("Random Forest Classifier\
                        \nBest F1-Score : 0.49\
                        \nBest Recall : 0.60")
                st.code("XGBoost Classifier\
                        \nBest F1-Score : 0.44\
                        \nBest Recall : 0.71")

with tab6:
    if st.session_state.cleaned_dataset is not None:
        st.header("Logistic Regression Evaluation")
        image_shap_one = Image.open("Pictures/shap_1.png")
        image_shap_two = Image.open("Pictures/shap_2.png")
        image_roc_curve = Image.open("Pictures/roc_curve.png")
        st.image(image_roc_curve, width=850)
        st.markdown("<br><br>",unsafe_allow_html=True)
        st.header("SHAP Plot")
        st.subheader("Machine 1")
        st.success("Stable")
        st.text("Despite Avg_Temperature trying to push the machine towards failure, the machine's maintenance history (Last_Service_Days) is good and the amount of stress its put under (Stress_Index) is pretty low. These two are pulling the machine back into the 'stable' zone.")
        st.image(image_shap_one, width=1000)
        st.markdown("<br><br>",unsafe_allow_html=True)
        st.subheader("Machine 2")
        st.error("High risk")
        st.text("The combined effect of heat and vibration levels (Heat_Plus_Vibration) and the amount of stress the machine is being put under (Stress_Index) is strongly pulling the machine towards failure. The maintenance history (Last_Service_Days) also appears to be bad. The positive effect of the other columns are getting overshadowed, as the signal from the negative ones are too strong.")
        st.image(image_shap_two, width=1000)

with tab7:
    st.header("Note regarding RUL predictions")
    st.text("The target column that needs to be used (Working_Hours_Total), has near zero correlation with any of the other columns in the dataset.")
    st.markdown("<br>",unsafe_allow_html=True)
    temp = st.session_state.cleaned_dataset.copy()
    for col in temp.select_dtypes(include=["object"]).columns:
        temp = pd.get_dummies(temp, columns=[col])

    correlation = temp.corr()["Working_Hours_Total"].sort_values()
    st.table(correlation)

    st.text("Because of the lack of correlation, models cannot properly derive patterns from the data to make predictions. Any prediction made by the current model should not be considered the most reliable.")
    st.markdown("---")
    st.text("The column Stress_Hours appears to have good correlation with target column, Working_Hours_Total, with a value of 0.58. But that's because Stress_Hours was derived using the target column itself with the following formula : ")
    st.code("df['Stress_Hours'] = df['Stress_Index'] * df['Working_Hours_Total']")
    st.text("Stress_Hours was excluded from the training and testing as including it would cause data leakage, which lead to misleading scores and predictions.")

with tab8:
    if st.session_state.cleaned_dataset is not None:
        st.header("Predict (Is it going to fail or will it survive)")
        ambient_humidity = st.slider("Select Ambient Humidity : ",0.0,st.session_state.cleaned_dataset["Ambient_Humidity"].max())
        st.markdown("<br><br>",unsafe_allow_html=True)
        avg_temp = st.slider("Select Average Temperature : ",0.0,st.session_state.cleaned_dataset["Avg_Temperature"].max())
        st.markdown("<br><br>",unsafe_allow_html=True)
        vib_level = st.slider("Select Vibration Level : ",0.0,15.0)
        st.markdown("<br><br>",unsafe_allow_html=True)
        voltage_fluctuation = st.slider("Select Voltage Fluctuation : ",0.0,st.session_state.cleaned_dataset["Voltage_Fluctuation"].max())
        st.markdown("<br><br>",unsafe_allow_html=True)
        torque = st.slider("Select Torque : ",0,100)
        st.markdown("<br><br>",unsafe_allow_html=True)
        last_service = st.slider("Select Last Service Days : ",0,st.session_state.cleaned_dataset["Last_Service_Days"].max())
        st.markdown("<br><br>",unsafe_allow_html=True)
        working_hours = st.slider("Select Total Working Hours : ",0,st.session_state.cleaned_dataset["Working_Hours_Total"].max())
        st.markdown("<br><br>",unsafe_allow_html=True)
    
        stress_index = vib_level * torque
        heat_plus_vibration = avg_temp * vib_level
        heat_plus_service = avg_temp * last_service

        input_list = [heat_plus_vibration, heat_plus_service, stress_index, avg_temp, last_service, ambient_humidity, torque, voltage_fluctuation, working_hours]

        input_list_rul = [heat_plus_vibration, heat_plus_service, stress_index, avg_temp, last_service, ambient_humidity, torque, voltage_fluctuation]

        clicked = st.button("Run Predictions")

        if clicked:
            failed, result = classification.predict(st.session_state.cleaned_dataset,input_list)
            rul = regression.predict(st.session_state.cleaned_dataset,input_list_rul,working_hours)
            if failed == 0:
                st.success(f"Stable : Probability of failure : {(result[0]*100):.2f}%")
                st.info(f"Remaining Life : {rul:.2f} hours")
            else:
                st.error(f"High Risk : Probability of failure : {(result[0]*100):.2f}%")
                st.info(f"Remaining Life : {rul:.2f} hours")




