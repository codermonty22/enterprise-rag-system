# 📚 Enterprise Document RAG System (AI & DS II Mini Project)

An offline Retrieval-Augmented Generation (RAG) system built with **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and **Streamlit** to query enterprise policy documents in real time.

---

## 🛠️ Tech Stack & Libraries
* **Framework**: LangChain (`langchain-community`, `langchain-huggingface`)
* **Vector Store**: ChromaDB
* **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **User Interface**: Streamlit
* **Language**: Python 3.11

---

## 📁 Repository Structure
```text
enterprise-rag-system/
├── data/
│   └── sample_policy.pdf          # Sample internal policy document
├── app.py                         # Streamlit Web UI (Real-time Demo)
├── model_pipeline.py              # Core RAG, Text Chunking & Retrieval Pipeline
├── eda_evaluation.ipynb           # EDA & Quantitative Model Metrics Notebook
├── requirements.txt               # Dependencies
├── .gitignore                     # Git exclusions
└── README.md                      # Project Documentation
```

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/aayushkoli/enterprise-rag-system.git](https://github.com/aayushkoli/enterprise-rag-system.git)
cd enterprise-rag-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
python -m streamlit run app.py
```

---

## 📊 Evaluation Metrics
* **Contextual Cosine Similarity**: ~0.88+
* **Precision@1 Retrieval Accuracy**: 100%
* **Embedding Dimension**: 384 dimensions