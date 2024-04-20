import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cardio_train.csv', sep=';')
df.drop('id', axis=1, inplace=True)


def data_preprocessing():
    st.header("Data Preprocessing")
    # drop Id and change age to years
    df['age'] = (df['age'] / 365).round().astype('int')
    df['height'] = df['height'] * 0.01

    # add subheader to explain the data
    st.subheader(
        "DataFrame after dropping Id and changing age to years and height to meters:")
    # Display the DataFrame
    st.write(df.head())
    st.write("Shape:", df.shape)

    # Add categories
    st.subheader(
        "Add rows to the DataFrame; BMI, Age Group, and Blood Pressure categories:")
    st.write("Calculate BMI and add to the DataFrame: Bmi = weight / (height ** 2)")
    df['bmi'] = (df['weight'] / (df['height'] ** 2)).astype('int')

    # Display the explanation to the user
    st.write("Explanation:")
    st.write(
        "We have added three new categories to the DataFrame based on existing columns:")
    st.write(
        "- BMI Category: Categorized into Underweight: 0, Normal: 1, Overweight: 2, and Obese: 3")
    st.write("- Age Group: Categorized into Young: 0, Middle-aged: 1, and Senior: 2")
    st.write("- Blood Pressure Category: Categorized into Normal: 0, Hypertension: 1, and High-Normal: 2")

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

    # Display the DataFrame
    st.write("Updated DataFrame:")
    st.write(df.head())

    # Define the list of features for visualization
    features = ['bmi', 'bmi_category', 'age_group',
                    'bp_category', 'cardio']

    # Sidebar
    selected_feature = st.sidebar.selectbox(
        "Select Feature to see Visualization of new features", features)

    # Title container
    with st.container():
        st.title(f"Visualization - {selected_feature} Distribution")

    # Visualize the distribution of the selected feature
    feature_counts = df[selected_feature].value_counts()
    st.bar_chart(feature_counts)
    
    with st.container():
        # st.title("Exploratory Data Analysis - Numeric Features")
        st.subheader("Correlation Matrix of new Numeric Features (Heatmap)")
    # features = ['bmi', 'bmi_category', 'age_group', 'bp_category', 'cardio', 'ap_hi', 'ap_lo']
    # Create a subset of the DataFrame with only numeric features
    numeric_df = df[features]
    # print(df)
    print(numeric_df)
    # numeric_df = df

    # Correlation matrix to see how features correlate with cardio
    plt.figure(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot(plt)


# data_preprocessing()
