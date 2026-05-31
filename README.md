# 📄 Document Chatbot – GenAI Domain Assistant with RAG & Semantic Search

**Course:** Introduction to Applied Artificial Intelligence
**Labs:** Week 9–12 (Project 3)
**Student:** Mansoor Ahmed
**Department:** BSCS
**Platform:** Python + OpenAI API + ChromaDB + Streamlit

---

## 📌 Overview

This project implements a **production-ready document-aware chatbot** that answers questions based on company documents using modern **Generative AI** techniques.

The system combines:

* 🤖 OpenAI GPT models for conversational AI
* 📚 Retrieval Augmented Generation (RAG)
* 🔍 Semantic Search using Embeddings & ChromaDB
* 🌐 Streamlit-based Web Interface

Unlike traditional keyword search, the chatbot understands context and synonyms. For example:

* **"time off"** → retrieves **vacation policy**
* **"WFH"** → retrieves **remote work policy**

This enables accurate, context-aware responses grounded in organizational documents.

---

## ✨ Key Features

| Feature                  | Description                                                      |
| ------------------------ | ---------------------------------------------------------------- |
| 💬 Conversational UI     | Interactive chat interface with conversation history             |
| 🔍 Semantic Search       | ChromaDB + OpenAI embeddings retrieve information by meaning     |
| 📚 RAG Pipeline          | Answers are generated using retrieved document context           |
| 🧠 Custom System Prompts | Supports HR assistant, IT support, customer support, etc.        |
| 🚀 Production Ready      | Session state, caching, error handling, and deployment support   |
| 📂 Document Agnostic     | Works with TXT files and can be extended to PDFs, DOCX, and HTML |
| ⚡ Fast Retrieval         | Vector database enables efficient semantic search                |
| 🌐 Web Application       | Built with Streamlit for easy use and deployment                 |

---

## 🛠️ Tech Stack

### Core Technologies

* Python 3.10+
* OpenAI API
* ChromaDB
* Streamlit
* LangChain
* python-dotenv

### Models

* GPT-3.5 Turbo / GPT-4
* text-embedding-ada-002

### Libraries

```text
openai
chromadb
langchain
langchain-community
streamlit
python-dotenv
pypdf
```

---

## 📁 Project Structure

```text
document-chatbot/
│
├── app.py                       # Main Streamlit application
├── .env                         # API keys (not committed)
├── requirements.txt             # Project dependencies
│
├── company_docs/
│   ├── hr_policy.txt
│   ├── benefits.txt
│   └── it_policy.txt
│
├── chroma_db/                   # Persistent vector database
│
└── notebooks/
    ├── week9_chatbot.ipynb
    ├── week10_rag_system.ipynb
    ├── week11_semantic_search.ipynb
    └── week12_streamlit_app.py
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/document-chatbot.git
cd document-chatbot
```

### 2️⃣ Create Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure API Key

Create a `.env` file:

```text
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

> ⚠️ Never commit your API key or `.env` file to GitHub.

### 5️⃣ Add Company Documents

Place text documents inside:

```text
company_docs/
```

Example:

**hr_policy.txt**

```text
Vacation Policy: Full-time employees receive 15 days of paid vacation per year.

Remote Work Policy:
Employees may work remotely up to 3 days per week with manager approval.

Parental Leave:
12 weeks of paid leave for primary caregivers.
```

---

## 🚀 Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open your browser:

```text
http://localhost:8501
```

---

## 💡 Example Questions

Try asking:

```text
How many vacation days do I get?

Can I work from home?

What is the parental leave policy?

Tell me about health insurance.

What benefits are available for employees?
```

The chatbot retrieves relevant document chunks and generates grounded answers using RAG.

---

# 🧠 Lab Achievements (Weeks 9–12)

---

## ✅ Week 9 – OpenAI Chatbot

### Objectives Completed

* Created OpenAI account and API key
* Made first GPT API call
* Built multi-turn conversation loop
* Implemented system prompts

### Skills Learned

* Prompt Engineering
* Chat Completion APIs
* Context Management
* Assistant Personalities

---

## ✅ Week 10 – RAG with Keyword Search

### Objectives Completed

* Loaded company documents
* Split documents into chunks
* Implemented keyword retrieval
* Built first RAG pipeline

### Key Concepts

* Retrieval Augmented Generation
* Context Injection
* Chunking Strategies
* Knowledge Grounding

### Outcome

The chatbot could answer questions using document content instead of relying solely on model knowledge.

---

## ✅ Week 11 – Semantic Search & Vector Databases

### Objectives Completed

* Generated embeddings
* Implemented cosine similarity
* Integrated ChromaDB
* Replaced keyword retrieval with semantic search

### Key Concepts

* Embeddings
* Vector Databases
* Similarity Search
* Semantic Understanding

### Example

| Query             | Retrieved Topic    |
| ----------------- | ------------------ |
| Time Off          | Vacation Policy    |
| WFH               | Remote Work Policy |
| Child Birth Leave | Parental Leave     |

### Outcome

The chatbot became significantly smarter and more flexible when retrieving information.

---

## ✅ Week 12 – Streamlit Deployment

### Objectives Completed

* Built complete Streamlit UI
* Added chat history
* Implemented session state
* Added caching for efficiency
* Added error handling
* Added sidebar controls

### Features Added

* Chat Interface
* Clear Chat Button
* Loading Spinners
* System Information Panel
* Document Statistics
* Deployment Ready Architecture

### Outcome

A fully functional web-based AI assistant ready for portfolio and deployment.

---

## 🧪 Example Interaction

### User

```text
Can I work from home?
```

### Assistant

```text
Yes. Employees may work remotely up to 3 days per week with manager approval according to the Remote Work Policy.
```

---

### User

```text
What about health insurance?
```

### Assistant

```text
Health insurance is fully covered for full-time employees. For detailed plan information, please contact HR.
```

---

### User

```text
How much does the company 401(k) match?
```

### Assistant

```text
The company provides 401(k) matching up to 5% of employee salary.
```

---

### Out-of-Scope Question

If information does not exist in the document collection:

```text
I don't have that information in my knowledge base.
Please contact HR for further assistance.
```

---

## 🔄 RAG Workflow

```text
User Question
      │
      ▼
Semantic Search
(ChromaDB)
      │
      ▼
Relevant Chunks Retrieved
      │
      ▼
Context Added to Prompt
      │
      ▼
OpenAI GPT Model
      │
      ▼
Grounded Answer Generated
```

---

## 📈 Future Improvements

* Support PDF documents
* Support DOCX documents
* Support HTML knowledge bases
* Add source citations
* Add reranking models
* Implement HyDE retrieval
* Multi-document collections
* User authentication
* Streamlit Cloud deployment
* Hugging Face Spaces deployment
* Conversation memory summarization
* Advanced analytics dashboard

---

## 🎯 Learning Outcomes

Through this project, the following AI concepts were successfully implemented:

* Generative AI
* Prompt Engineering
* OpenAI APIs
* Retrieval Augmented Generation (RAG)
* Embeddings
* Vector Databases
* Semantic Search
* Streamlit Application Development
* AI System Deployment

---

## 👨‍💻 Author

**Mansoor Ahmed**
BSCS – Applied Artificial Intelligence

---

