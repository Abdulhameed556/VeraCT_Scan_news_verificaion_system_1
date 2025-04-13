import streamlit as st
import requests

st.title("AI-Powered News Verification")
st.write("Check if a news headline and content are truthful based on reliable sources.")

# Input fields
headline = st.text_input("News Headline", "")
content = st.text_area("News Content", "")

if st.button("Verify News"):
    if not headline or not content:
        st.warning("Please enter both a headline and content.")
    else:
        api_url = "http://127.0.0.1:8000/verify"  # Update if deployed
        payload = {"headline": headline, "content": content}
        response = requests.post(api_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            st.subheader("Verification Result")
            st.write(f"Verdict: **{result['verdict']}**")
            st.write(f"LLM Analysis: {result['llm_response']}")
        else:
            st.error("Error verifying news. Please check the input and try again.")
