import streamlit as st
import pandas as pd

from analysis_visualizations import analysis_visualizations
from data_preprocessing import data_preprocessing
from Data_splitting import data_splitting
from Model_training import model_building
from Model_testing import model_testing

# Title container
with st.container():
    st.title("Machine Learning Model to test for cardio Vascular Diseases")

    functions = [analysis_visualizations,data_preprocessing, data_splitting,  model_building, model_testing]
    
    # Initialize session state for the index of the current graph
    if 'current_function_index' not in st.session_state:
        st.session_state.current_function_index = 0
    
    # Display the current graph
    current_function_index = st.session_state.current_function_index
    
    for i in range(len(functions)):
        if i == current_function_index:
            functions[i]()
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.current_function_index > 0:
            if st.button('Previous'):
                st.session_state.current_function_index -= 1
                st.experimental_rerun()
    
    with col2:
        if st.session_state.current_function_index < len(functions) -1:
            if st.button('Next'):
                st.session_state.current_function_index += 1
                st.experimental_rerun()
