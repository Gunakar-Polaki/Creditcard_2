import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Define the Streamlit app
def main():
    st.title("Email Spam Detection")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load the trained model and vectorizer
            model = joblib.load('C:\\Users\\LENOVO\\Downloads\\spam_detection_model.sav')
            vectorizer = joblib.load('C:\\Users\\LENOVO\\Downloads\\spam_vectorizer.sav')

            data = pd.read_csv(uploaded_file)

            # Preprocess the data
            X = vectorizer.transform(data['Message'])

            # Make predictions
            predictions = model.predict(X)

            # Display results
            results = pd.DataFrame({'Message': data['Message'], 'Category': predictions})
            st.write(results)

        except Exception as e:
            st.error("Error loading the model or vectorizer. Please make sure the files are correct and not corrupted.")

# Run the app
if __name__ == '__main__':
    main()
