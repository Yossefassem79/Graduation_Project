# 🤖 AI-Powered Graduation Project Recommendation System

## 📌 Overview

This project implements an intelligent AI-powered recommendation and semantic similarity platform for graduation projects using:

* Natural Language Processing (NLP)
* Semantic Search
* Vector Embeddings
* Hybrid Ranking Systems
* Large Language Models (LLMs)

The system helps students:

* discover unique graduation project ideas
* avoid duplicate projects
* analyze originality
* generate intelligent project features
* receive context-aware recommendations through an AI chatbot

---

# ⚙️ System Pipeline

## 1️⃣ Data Preprocessing

* Text normalization
* Duplicate removal
* Smart content merging
* Technical keyword extraction
* Feature engineering

## 2️⃣ Feature Extraction

* KeyBERT-based keyword extraction
* Automatic technical term detection
* Semantic feature generation

## 3️⃣ Embedding Generation

* SentenceTransformer embeddings
* Normalized vector representations
* Semantic encoding of projects

## 4️⃣ Semantic Retrieval

* FAISS vector indexing
* Nearest-neighbor semantic search
* Fast project similarity lookup

## 5️⃣ Hybrid Ranking

The final ranking combines:

* Semantic similarity
* Feature similarity
* Coverage ratio
* Confidence estimation
* Originality analysis

## 6️⃣ AI Recommendation Engine

* Context-aware project generation
* Feature recommendation
* Novelty checking
* Conversational chatbot assistance

---

# 🧠 AI & NLP Technologies Used

## 🔹 Machine Learning & NLP

* SentenceTransformers
* KeyBERT
* Scikit-learn
* SciPy
* FAISS

## 🔹 LLM Integration

* Google Gemini API
* Ollama
* Mistral

## 🔹 Backend & Infrastructure

* FastAPI
* Pandas
* NumPy
* Python

---

# 🏗️ Project Architecture

```text
User Query
    ↓
Intent Classification
    ↓
Context Builder
    ↓
Feature Extraction
    ↓
Embedding Generation
    ↓
FAISS Semantic Search
    ↓
Hybrid Ranking Engine
    ↓
Originality & Duplicate Analysis
    ↓
AI Recommendation Response
```

---

# 🔍 Similarity Engine Workflow

```text
Raw Dataset
    ↓
Preprocessing
    ↓
Feature Extraction
    ↓
Sentence Embeddings
    ↓
FAISS Indexing
    ↓
Semantic Retrieval
    ↓
Feature Similarity Matching
    ↓
Hybrid Re-ranking
    ↓
Final Recommendation
```

---

# 🚀 Features

## ✅ AI Chatbot

* Context-aware conversations
* Intent classification
* Domain-specific recommendations
* Memory-aware responses

## ✅ Semantic Similarity Search

* Embedding-based retrieval
* Semantic duplicate detection
* Vector search with FAISS

## ✅ Hybrid Recommendation System

* Multi-stage ranking pipeline
* Feature-level semantic comparison
* Adaptive scoring strategy

## ✅ Originality Detection

* Duplicate risk analysis
* Originality scoring
* Similarity confidence estimation

## ✅ Intelligent Feature Generation

* AI-generated project features
* Novelty-aware generation
* Domain-aware recommendations

---

# 📊 Evaluation

The system includes:

* Self-retrieval evaluation
* Real-query testing
* Hybrid ranking validation
* Confidence scoring

### Evaluation Metrics

* Semantic Similarity Score
* Hybrid Score
* Originality Score
* Confidence Score
* Duplicate Risk Classification

---

# 📁 Project Structure

```text
GRADUATION_PROJECT/
│
├── api/                         # FastAPI backend
│
├── Data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned dataset
│
├── models/                     # FAISS index & metadata
│
├── Notebooks/
│   └── TEST.ipynb              # Training & evaluation notebook
│
├── src/
│   ├── recommendation_engine/  # Chatbot & recommendation logic
│   └── similarity_model/       # Semantic search engine
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🧩 Recommendation Engine Modules

## recommendation_engine/

Contains:

* Chatbot engine
* Intent classification
* Prompt building
* Idea generation
* Feature generation
* Memory management
* Novelty checking
* Response formatting

---

# 🔬 Similarity Model Modules

## similarity_model/

Contains:

* Semantic search
* Embedding engine
* Hybrid ranker
* Feature similarity engine
* Preprocessing pipeline
* Evaluation framework

---

# ⚡ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

# ▶️ Running The Project

## Run FastAPI Server

```bash
uvicorn api.main:app --reload
```

---

## Run Notebook

```bash
jupyter notebook
```

Open:

```text
Notebooks/TEST.ipynb
```

---

# 💡 Example Query

## Input

```text
AI-based smart library recommendation platform
```

## Output

* Similar graduation projects
* Semantic similarity scores
* Originality analysis
* Duplicate risk estimation
* Recommended features

---

# 🎯 Future Improvements

* Full RAG integration
* Multi-agent orchestration
* GPU acceleration
* Advanced evaluation metrics
* Real-time deployment
* Database persistence
* Frontend dashboard

---

# 📚 Research Areas Covered

* Natural Language Processing (NLP)
* Semantic Search
* Recommendation Systems
* Vector Databases
* Conversational AI
* Information Retrieval
* Hybrid Ranking Systems
* Large Language Models (LLMs)

---

# 👨‍💻 Author

Yossef Assem

---

# 📄 License

This project is for educational and research purposes.
