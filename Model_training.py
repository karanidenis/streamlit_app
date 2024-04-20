import streamlit as st
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import LearningRateScheduler
from contextlib import redirect_stdout
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import Sequential
# import pickle
import joblib



def data_preprocessing():
    df = pd.read_csv('cardio_train.csv', sep=';')
    df.drop('id', axis=1, inplace=True)
    df['age'] = (df['age'] / 365).round().astype('int')
    df['height'] = df['height'] * 0.01

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

    return df


processed_df = data_preprocessing()

# st.title("Model Training")

# st.write("Shape:", processed_df.shape)

X = processed_df.drop('cardio', axis=1).to_numpy()
y = processed_df['cardio'].to_numpy()

# data Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Further Splitting
x_train, x_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)


# Model Building

def model_building():

    def model(input_dim):
        st.header("Neaural Networks Model Building and Training")
        # st.title("Neaural Networks Model Building and Training")
        st.write("Data Splitting and Processing Completed")
        st.code("""model = Sequential([
            Dense(500, input_dim=x_train.shape[1], activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss=binary_crossentropy,
                    optimizer=Adam(learning_rate=0.0001),
                    metrics=['accuracy'])
        return model""")
        
        model = Sequential([
            Dense(500, input_dim=input_dim, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss=binary_crossentropy,
                    optimizer=Adam(learning_rate=0.0001),
                    metrics=['accuracy'])
        return model
    
    input_dim = x_train.shape[1]
    model = model(input_dim)

    # Function to display model summary


    def display_model_summary(model):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            model.summary()
        model_summary_text = buffer.getvalue()
        st.text("Model Summary:")
        st.text(model_summary_text)


    # model = model()
    display_model_summary(model)

    # Button to initiate training
    st.code("""def train(model, x_train, y_train, x_val, y_val, X_test, y_test):
        history = model.fit(x_train, y_train, validation_data=(
            x_val, y_val), epochs=20, batch_size=64)
        score = model.evaluate(X_test, y_test, verbose=0)
        return history, score
            """)
    
    def train(model, x_train, y_train, x_val, y_val, X_test, y_test):
        history = model.fit(x_train, y_train, validation_data=(
            x_val, y_val), epochs=20, batch_size=64)
        # score = model.evaluate(X_test, y_test, verbose=0)
        return history
    

    def save(model):
        # Save the model architecture as JSON
        model_json = model.to_json()
        with open("saved_model.json", "w") as json_file:
            json_file.write(model_json)
        
        # Save the model weights
        model.save_weights("saved_model_weights.weights.h5")
        st.write("Model saved")

    if st.button("Train Model"):
        st.write("Training the model...")
        history = train(model, x_train, y_train, x_val, y_val, X_test, y_test)
        # score = model.evaluate(X_test, y_test, verbose=0)
        st.write("Model Training Completed")
        st.text("Evaluating the model on test data...")
        
        # Display evaluation results
        st.subheader("Evaluation Results")

        # Evaluate the model
        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

        # Plot training history
        st.subheader("Training History")
        sns.lineplot(x=range(1, len(history.history['loss']) + 1), y=history.history['loss'], label='train')
        sns.lineplot(x=range(1, len(history.history['val_loss']) + 1), y=history.history['val_loss'], label='test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        st.pyplot(plt)


        # Evaluate the model on test data
        st.subheader("Model Evaluation on Test Data")
        st.write("Evaluating the model...")
        score = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Test loss: {score[0]:.2f}%")
        st.write(f"Test accuracy: {score[1]:.2f}%")
            
        model_instance = model
        save(model_instance)

    # return train
# model_building()