import streamlit as st
import pandas as pd
import pickle

# Define the Streamlit app
def main():
    st.title("Credit Card Fraud Detection")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    diagnosis = ""
    if uploaded_file is not None:
        try:
            # Load the trained model
            model = pickle.load(open('trained_model.sav', 'rb'))

            data = pd.read_csv(uploaded_file)

            # Make predictions
            predictions = model.predict(data)

            if predictions[0] == 0:
                diagnosis = 'The user is a Valid User'
            else:
                diagnosis = 'The user is an Invalid User'

        except Exception as e:
            st.error("Error loading the model. Please make sure the file is correct and not corrupted.")
    st.success(diagnosis)

# Run the app
if __name__ == '__main__':
    main()
