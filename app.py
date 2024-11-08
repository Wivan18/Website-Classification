import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
import h5py
from urllib.parse import urlparse

# Load the Logistic Regression model saved in HDF5 format
def load_logistic_model(file_path):
    with h5py.File(file_path, 'r') as h5file:
        coefficients = h5file['coefficients'][:]
        intercept = h5file['intercept'][()]
        
    # Create a Logistic Regression model with loaded parameters
    model = LogisticRegression()
    model.coef_ = np.array([coefficients])  # Must be 2D array
    model.intercept_ = np.array([intercept])  # Must be 1D array
    model.classes_ = np.array([0, 1])  # Binary classes
    return model

# Load the model globally
model = load_logistic_model('D:/Data_Science_Projects/Website_Classification/logistic_regression_model.h5')

def welcome():
    return "Welcome to the Website Checker App!"

def extract_features(url):
    """
    Extract features from the URL for classification.
    
    Args:
        url (str): The URL to extract features from.
    
    Returns:
        np.array: Extracted features as a NumPy array.
    """
    parsed_url = urlparse(url)
    
    # Example feature extraction (you can adjust these as needed)
    features = [
        len(url),                  # Length of the URL
        len(parsed_url.netloc),    # Length of the domain name
        len(parsed_url.path),       # Length of the path
        1 if parsed_url.scheme == "https" else 0  # 1 if HTTPS, 0 if not
        # Add more features as needed
    ]
    
    return np.array(features).reshape(1, -1)  # Reshape for the model

def predict_url_classification(url):
    """
    Predict the classification of a URL based on input features.
    
    Args:
        url (str): The URL to be predicted.
    
    Returns:
        str: Predicted label (0 for phishing, 1 for legitimate).
    """
    features = extract_features(url)  # Extract features from the URL
    prediction = model.predict(features)
    return "Legitimate URL" if prediction[0] == 1 else "Phishing URL"

def main():
    st.title('URL Classification Predictor')
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">URL Classification Predictor</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input field for the URL
    url_input = st.text_input("Enter a URL to classify")

    # Predict the class if the user clicks the Predict button
    if st.button("Predict"):
        if url_input:
            result = predict_url_classification(url_input)
            st.success(f'The predicted classification is: {result}')
        else:
            st.error("Please enter a valid URL.")

    # About section
    if st.button("About"):
        st.text("URL Classification App")
        st.text("Built with Streamlit and Logistic Regression")

if __name__ == '__main__':
    main()
