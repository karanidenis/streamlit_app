# Title container
import io
from contextlib import redirect_stdout
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv(
    'cardio_train.csv', sep=';')
# df.drop('id', axis=1, inplace=True)

def analysis_visualizations():
    with st.container():
        st.header("Data analysis and Visualization")
        st.subheader("This is the data set for the model")
    
    st.sidebar.subheader("Display Data")
    rows_to_display = st.sidebar.slider(
        "Number of Rows", min_value=1, max_value=100, value=5)
    st.write(df.head(rows_to_display))
    st.write("Shape:", df.shape)


    with st.container():
        # st.title("Exploratory Data Analysis - Numeric Features")
        st.subheader("Correlation Matrix of Numeric Features (Heatmap)")
    features = df.columns
    # Create a subset of the DataFrame with only numeric features
    numeric_df = df[features]
    # print(df)
    numeric_df = df

    # Correlation matrix to see how features correlate with cardio
    plt.figure(figsize=(8, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot(plt)
    
    # Define the list of features for visualization
    features = ['age', 'weight', 'height', 'cholesterol', 'gender', 'gluc', 'active', 'smoke', 'alco', 'cardio', 'ap_hi', 'ap_lo']
    df['age'] = (df['age'] / 365).round().astype('int')

    # Sidebar
    selected_feature = st.sidebar.selectbox("Select Feature to see Visualization", features)

    # Title container
    with st.container():
        st.title(f"Visualization - {selected_feature} Distribution")

    # # Check if the selected feature is 'age', convert it to years if so
    # if selected_feature == 'age':
    #     df[selected_feature] = (df[selected_feature] / 365).round().astype('int')

    # Visualize the distribution of the selected feature
    feature_counts = df[selected_feature].value_counts()
    st.bar_chart(feature_counts)
