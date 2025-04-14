import streamlit as st
import requests

st.set_page_config(page_title="AI News Verifier", layout="centered")

st.title("📰 AI-Powered News Verification")
st.write("Check if a news headline and content are truthful based on reliable sources using AI.")

# Input fields
headline = st.text_input("🗞️ News Headline", "")
content = st.text_area("🧾 News Content", "")

if st.button("🔍 Verify News"):
    if not headline or not content:
        st.warning("⚠️ Please enter both a headline and content.")
    else:
        api_url = "https://news-web-application-ai-news-detector.onrender.com/verify"  # <- Make sure this matches your FastAPI route
        
        payload = {"headline": headline, "content": content}
        
        try:
            response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ News verification completed.")
                st.subheader("🧠 AI Verdict")
                st.markdown(f"**Verdict:** {result['verdict']}")
                st.markdown("**LLM Analysis:**")
                st.write(result['llm_response'])
            else:
                st.error(f"❌ Server responded with status code {response.status_code}. Please try again.")
        
        except requests.exceptions.RequestException as e:
            st.error(f"🚫 Request failed: {e}")
