import pandas as pd
import numpy as np
import streamlit as st
import io
import train_model
import regression
import preprocess_data

st.set_page_config(layout="wide")
st.title("Predictive Maintenance")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload dataset","Preprocessed data","Process","Train models","Predict"])

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
        st.header("Train models")
        clicked = st.button("Train models (Logistic Regression, RandomForestClasifier, XGBoostClassifier)")

        if clicked:
            plot, report = train_model.train_classification(1,st.session_state.cleaned_dataset)

            st.pyplot(plot)
            st.markdown("<br><br>",unsafe_allow_html=True)
            st.code(report)

with tab5:
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
            failed, result = train_model.predict(st.session_state.cleaned_dataset,input_list)
            rul = regression.predict(st.session_state.cleaned_dataset,input_list_rul,working_hours)
            if failed == 0:
                st.success(f"Stable : Probability of failure : {(result[0]*100):.2f}")
                st.info(f"Remaining Life : {rul}")
            else:
                st.error(f"High Risk : Probability of failure : {(result[0]*100):.2f}")
                st.info(f"Remaining Life : {rul}")


