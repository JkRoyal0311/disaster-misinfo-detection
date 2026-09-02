import streamlit as st
import joblib
import sys
import os

# Add the parent directory to the path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocess import clean_text

# Page config
st.set_page_config(page_title="Disaster Misinfo Detector", page_icon="🚨")

st.title("🚨 Disaster Misinformation Detector")
st.write("Enter a tweet or news headline below to check if it's real or fake.")

# Cache the model loading so it doesn't reload every time you click a button
@st.cache_resource
def load_models():
    model_path = os.path.join(os.path.dirname(__file__), '../models/logistic_model.pkl')
    vec_path = os.path.join(os.path.dirname(__file__), '../models/tfidf_vectorizer.pkl')
    
    if os.path.exists(model_path) and os.path.exists(vec_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
        return model, vectorizer
    return None, None

# Load the models
model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.error("⚠️ Error: Models not found. Please run the Week 2 notebook first!")
else:
    user_input = st.text_area("Enter news text here:", height=150)

    if st.button("Analyze Text"):
        if user_input.strip():
            # 1. Clean the text using your reusable function
            cleaned = clean_text(user_input)
            st.write(f"**Cleaned Text seen by Model:** `{cleaned}`")
            
            # 2. Convert text to numbers using the saved TF-IDF vectorizer
            vectorized_text = vectorizer.transform([cleaned])
            
            # 3. Make the prediction
            prediction = model.predict(vectorized_text)[0]
            probabilities = model.predict_proba(vectorized_text)[0]
            
            # 4. Display the results with confidence scores
            st.markdown("---")
            if prediction == 1: # 1 was encoded as 'fake'
                st.error(f"🛑 **Prediction: FAKE NEWS**")
                st.write(f"Confidence: **{probabilities[1]*100:.1f}%**")
            else:               # 0 was encoded as 'real'
                st.success(f"✅ **Prediction: REAL NEWS**")
                st.write(f"Confidence: **{probabilities[0]*100:.1f}%**")
        else:
            st.warning("Please enter some text to analyze.")