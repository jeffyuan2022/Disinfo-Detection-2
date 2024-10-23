import streamlit as st
import google.generativeai as genai
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")

# Gemini API setup
genai.configure(api_key=)
model = genai.GenerativeModel('gemini-1.5-flash-002')

# Streamlit app
st.title("Factuality Factor Report")

# Define factuality factors
factuality_factors = [
    "Biases",
    "ClickBait",
    "Confirmation Bias",
    "Context Veracity",
    "Information Utility"
]

# User inputs
statement = st.text_area("Enter the statement to analyze:")
selected_factors = st.multiselect("Select factuality factors to analyze:", factuality_factors)

def extract_analysis(text):
    analysis = {}
    factor_pattern = r'(\w+(?:\s+\([^)]+\))?)\s*:\s*Score:\s*(\d+)/10\s*Explanation:\s*(.+?)(?=\n\w+:|$)'
    matches = re.findall(factor_pattern, text, re.DOTALL)
    for match in matches:
        factor, score, explanation = match
        analysis[factor.strip()] = {
            "score": int(score),
            "explanation": explanation.strip()
        }
    return analysis

if st.button("Analyze Statement"):
    if statement and selected_factors:
        # Construct the prompt for Gemini
        prompt = f"""
        Analyze the following statement based on the selected factuality factors in the context of misinformation/disinformation:
        
        Statement: "{statement}"
        
        Factors to analyze: {', '.join(selected_factors)}
        
        For each factor, provide a score between 0 to 1 (where 0 is lowest and 1 is highest) with 2 decimals and a brief explanation.
        
        Format your response as follows for each factor as a dictionary:
        Factor(bolded) /n
        Score: /n
        Explanation: Brief explanation here
        
        Ensure each factor analysis is separated by a newline.
        """
        
        # Generate response from Gemini
        response = model.generate_content(prompt)
        

        # Display results
        st.subheader("Analysis Results")
        if response:
            st.write(response.text)
            st.write("---")
        else:
            st.error("Failed to analyze factors")
    else:
        st.warning("Please enter a statement and select at least one factor to analyze.")

# Biases
# Confirmation Bias
# Context Veracity
# Information Utility