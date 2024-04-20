import pandas as pd 
import numpy as np 
import streamlit as st 
from sklearn.preprocessing import StandardScaler

from Model_training import model_building


from keras.models import model_from_json

def load_model():
    # Load model architecture from JSON file
    with open('saved_model.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
    
    # Load model weights
    loaded_model.load_weights("saved_model_weights.weights.h5")
    
    return loaded_model
saved_model = load_model()


def model_testing():
    def welcome(): 
        st.title('welcome all')

    welcome()
    
    # defining the function which will make the prediction using  

    def preprocess_data(age, weight, height, cholesterol, gender, gluc, active, smoke, alco, ap_hi, ap_lo):
        test_data = {
            'age': [age],
            'gender': [gender],
            'height': [height],
            'weight': [weight],
            'ap_hi': [ap_hi],
            'ap_lo': [ap_lo],
            'cholesterol': [cholesterol],
            'gluc': [gluc],
            'smoke': [smoke],
            'alco': [alco],
            'active': [active]
        }
        df = pd.DataFrame(test_data)
        
        # Convert specific columns to integers
        int_columns = ['age', 'weight', 'height', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        df[int_columns] = df[int_columns].astype(int)

        # Data preprocessing
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
        bmi_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}
        df['bmi_category'] = df['bmi_category'].map(bmi_mapping)

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

        df['bp_category'] = df.apply(lambda row: categorize_bp(row['ap_hi'], row['ap_lo']), axis=1)
        bp_mapping = {'Normal': 0, 'Hypertension': 1, 'High-Normal': 2}
        df['bp_category'] = df['bp_category'].map(bp_mapping)

        print(df.columns)
        return df

    def prediction(numeric_df):   
    
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_df)
        model = saved_model
        y_pred = model.predict(scaled_features)
        class_label = (y_pred >= 0.5).astype(int)
        
        return class_label[0][0]

    # this is the main function in which we define our webpage  
    def main(): 
        # giving the webpage a title 
        st.title("Cardio Vascular Prediction") 
        
        age = st.number_input("Patient's Age in years", min_value=0, max_value=120, value=30)
        weight = st.number_input("Patient's Weight in kgs", min_value=0, max_value=500, value=70)
        height = st.number_input("Patient's Height in cms", min_value=0, max_value=300, value=170)
        gender_options = {"Male": 0, "Female": 1}
        genders = st.selectbox("Patient's Gener", list(gender_options.keys()))
        gender = gender_options[genders]
        gluc = st.selectbox("Patient's Glucose Levels", [0, 1, 2])
        activity_options = {"Not physically active": 0, "Active": 1}
        activity = st.selectbox("Patient's Activity Level", list(activity_options.keys()))
        # Retrieve the numerical value corresponding to the selected option
        active = activity_options[activity]
        
        smoke_options = {"Doesn't smoke": 0, "Active smoker": 1}
        smoking = st.selectbox("Patient's Smoking Habits", list(smoke_options.keys()))
        smoke = smoke_options[smoking]
        cholesterol = st.selectbox("Patient's Cholesterol Levels", [1, 2, 3])
        # alco = st.selectbox("Patient's Alcohol Habits", ["Non-alcoholic", "Alcoholic"])
        alcohol_options = {"Non-alcoholic": 0, "Alcoholic": 1}
        alcohol = st.selectbox("Patient's Alcohol Habits", list(alcohol_options.keys()))
        # Retrieve the numerical value corresponding to the selected option
        alco = alcohol_options[alcohol]
        ap_hi = st.number_input("Patient's Diastolic Blood Pressure", min_value=0, max_value=300, value=120)
        ap_lo = st.number_input("Patient's Systolic Blood Pressure", min_value=0, max_value=300, value=80)
        result = ""
        
        # the below line ensures that when the button called 'Predict' is clicked,  
        
        if st.button("Predict"): 
            numeric_df = preprocess_data(age, weight, height, cholesterol, gender, gluc, active, smoke, alco, ap_hi, ap_lo)
            result = prediction(numeric_df)
            
            # Additional info
            bmi = calculate_bmi(weight, height)
            age_category = categorize_age(int(age))
            
            # Output categorization
            if result == 0:
                prediction_result = "Less likely"
            else:
                prediction_result = "More likely"
            
            # Display result with additional info
            st.success(f"The output is {prediction_result}.\nBMI: {bmi}, Age Category: {age_category}")
        
    main()

 # BMI calculation function
def calculate_bmi(weight, height):
    bmi = float(weight) // ((float(height) / 100) ** 2)
    return bmi

# Age categorization function
def categorize_age(age):
    if age < 40:
        return 'Young'
    elif 40 <= age < 60:
        return 'Middle-aged'
    else:
        return 'Senior'
        
        
# model_testing()