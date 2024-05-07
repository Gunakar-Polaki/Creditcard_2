import streamlit as st
import pandas as pd
import pickle

# Define the Streamlit app
def main():
    st.title("Credit Card Fraud Detection")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Load the trained model
            model = pickle.load(open('trained_model_2.sav', 'rb'))

            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Assume Excel file
                data = pd.read_excel(uploaded_file)

            # Make predictions for each record
            predictions = model.predict(data)

            # Display predictions
            st.write("Predictions:")
            for i, prediction in enumerate(predictions):
                if prediction == 0:
                    st.write(f"Record {i+1}: The user is a Valid User")
                else:
                    st.write(f"Record {i+1}: The user is an Invalid User")

        except Exception as e:
            st.error(f"Error loading the model or processing the data: {e}")

# Run the app
if __name__ == '__main__':
    main()
