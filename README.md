# 🧠 VeraCT Scan-like Fake News Detection System

A powerful, AI-enhanced news verification API that analyzes the credibility of news content by cross-referencing reliable sources, extracting entities, and applying reasoning from large language models. Built with FastAPI, LangChain, Groq's LLaMA-4 model, spaCy, and multiple real-time search APIs.

---

## 🚀 Key Features

- ✅ **Headline & Content Verification** using LLM + multi-source matching
- 🌐 **Reliable Domain Detection** from over 600+ trusted sources
- 🧠 **LLM Verdict Generation** (VERIFIED, LIKELY FAKE, or REQUIRES MORE VERIFICATION)
- 🔍 **Entity Extraction** with spaCy (ORG, PERSON, EVENT, etc.)
- 🛡️ **Cross-checking** with [SearchAPI](https://www.searchapi.io/) and [Tavily](https://www.tavily.com/)
- 🔐 **.env-based API Key Security**
- 🧪 API Testing endpoints for dev setup

---

## 🧩 Tech Stack

- **Python 3.10+**
- **FastAPI**
- **LangChain + Groq LLaMA-4**
- **spaCy (NER & NLP)**
- **SearchAPI + Tavily**
- **Pydantic + CORS middleware**
- **Uvicorn (ASGI server)**

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Abdulhameed556/VeraCT_Scan-like_news_verification_system_1.git
cd VeraCT_Scan-like_News_Verification_System
