from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cardio_train.csv', sep=';')
df.drop('id', axis=1, inplace=True)
df['age'] = (df['age'] / 365).round().astype('int')
df['height'] = df['height'] * 0.01
def data_preprocessing(df):

    df['bmi'] = (df['weight'] / (df['height'] ** 2)).astype('int')

    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 25:
            return 'Normal'
        elif 25 <= bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'

    df['bmi_category'] = df['bmi'].apply(categorize_bmi)
    bmi_mappping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
    df['bmi_category'] = df['bmi_category'].map(bmi_mappping)

    def categorize_age(age):
        if age < 40:
            return 'Young'
        elif 40 <= age < 60:
            return 'Middle-aged'
        else:
            return 'Senior'

    df['age_group'] = df['age'].apply(categorize_age)
    age_mapping = {'Young': 0, 'Middle-aged': 1, 'Senior': 2}
    df['age_group'] = df['age_group'].map(age_mapping)

    def categorize_bp(ap_hi, ap_lo):
        if ap_hi < 120 and ap_lo < 80:
            return 'Normal'
        elif ap_hi >= 140 or ap_lo >= 90:
            return 'Hypertension'
        else:
            return 'High-Normal'

    df['bp_category'] = df.apply(lambda row: categorize_bp(
        row['ap_hi'], row['ap_lo']), axis=1)
    bp_mapping = {'Normal': 0, 'Hypertension': 1, 'High-Normal': 2}
    df['bp_category'] = df['bp_category'].map(bp_mapping)

    features = ['bmi', 'bmi_category', 'age_group', 'bp_category', 'cardio']
    numeric_df = df[features]

    return df


processed_df = data_preprocessing(df)

def data_splitting():
    # Title container
    with st.container():
        st.header("Data Splitting")
        st.subheader("This is the preprocessed data set for the model")

        st.write(processed_df.head())
        st.write("Shape:", processed_df.shape)

        # generate 2d classification dataset
        st.subheader("Data splitting and processing")
        st.code("X = df.drop('cardio', axis=1).to_numpy()")
        X = processed_df.drop('cardio', axis=1).to_numpy()

        st.write("X[:2]", X[:2])
        st.write("X.shape:", X.shape)

        st.code("y = df['cardio'].to_numpy()")
        y = processed_df['cardio'].to_numpy()
        st.write("y[:2]", y[:2])
        st.write('y.shape:', y.shape)

        # data Scaling
        st.subheader("Scaling the data...")
        st.code("""scaler = StandardScaler()""")
        scaler = StandardScaler()

        st.code("X = scaler.fit_transform(X)")
        X = scaler.fit_transform(X)
        st.write("X after scaling")
        st.code("X[:2]")
        st.write(X[:2])

        # Data Splitting
        st.subheader("Splitting the data into train and test sets...")
        st.code(
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.write("- X_train.shape:", X_train.shape)
        st.write("- X_test.shape:", X_test.shape)
        st.write("- y_train.shape:", y_train.shape)
        st.write("- y_test.shape:", y_test.shape)

        # Further Splitting
        st.write(
            "Further splitting the training data into training and validation sets...")
        st.code("x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)")
        x_train, x_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        st.write("- x_train.shape:", x_train.shape)
        # st.write("- x_train[:2]", x_train[:2])
        st.write("- x_val.shape:", x_val.shape)
        st.write("- y_train.shape:", y_train.shape)
        st.write("- y_val.shape:", y_val.shape)
        
