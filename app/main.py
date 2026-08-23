import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text

st.title("🚨 Disaster Misinformation Detector")
st.write("Enter a tweet or news headline below to check if it's real or fake.")

user_input = st.text_area("Enter news text here:")

if st.button("Analyze Text"):
    if user_input:
        # 1. Clean the text using your reusable function
        cleaned = clean_text(user_input)
        st.write(f"**Cleaned Text seen by Model:** {cleaned}")
        
        # 2. Placeholder for your actual Week 2/3 model
        st.info("Prediction: (Model will connect here in Week 4!)")
    else:
        st.warning("Please enter some text to analyze.")