---
title: Minirag
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---
# Mini RAG
mini rag which extracts information as per user query. 
resume = ( https://drive.google.com/file/d/1KmFouzTYfeRR_U8BX3IFLO3gU48fITEE/view?usp=sharing )
``` 
+--------------------------- Docker Container -----------------------------+
|                                                                       |
|  +-------------+          +----------------+        +--------------+   |
|  |  Streamlit  |--------->| RAG Pipeline  |<------>|   Storage    |   |
|  |  Web App    |          |               |        |   Layer      |   |
|  | (port:8501) |          +----------------+        +--------------+   |
|  +-------------+                  |                                    |
|        ^                         |                                    |
|        |                         v                                    |
|  +-----------+          +----------------+        +--------------+    |
|  |   API     |          |    Models     |        |  Document    |    |
|  |  Layer    |<-------->|   - LLM       |<------>| Processing   |    |
|  |           |          |   - Embeddings |        |              |    |
|  +-----------+          +----------------+        +--------------+    |
|                                                                      |
+----------------------------------------------------------------------+
```
External User ---(HTTP/8501)---> Docker Container

Flow:
1. User -> Streamlit Interface
2. Streamlit -> RAG Pipeline
3. RAG Pipeline <-> Document Processing
4. RAG Pipeline <-> Models (LLM/Embeddings)
5. Models <-> Storage Layer
6. RAG Pipeline -> API Layer
7. API Layer -> User Response

Key Components:
- Entry Point: Streamlit Web Application
- Core: RAG Pipeline
- Processing: Document Handler, Text Splitter
- Storage: Vector Store, Document Storage
- Models: LLM, Embedding Models
- Infrastructure: Docker, Health Checks

## INDEX CONFIG 

### 1. Document Processing
- **Chunking Strategy:**
  - **Method:** `RecursiveCharacterTextSplitter` from LangChain.
  - **Parameters:**
    - `chunk_size`: 1000 characters
    - `chunk_overlap`: 150 characters
    - `separators`: `["\n\n", "\n"]`

### 2. Embedding Generation
- **Provider:** [Cohere](https://cohere.com/)
- **Model:** `embed-english-v3.0`
- **Vector Dimension:** 1024

### 3. Vector Database & Indexing

- **Provider:** [Pinecone](https://www.pinecone.io/)
- **Index Configuration:**
  - **Name:** `miniragindex`
  - **Metric:** `cosine`
  - **Specification:** `ServerlessSpec`
  - **Cloud Provider:** `aws`
  - **Cloud Region:** `us-east-1`
- **Data Schema (per vector):**
  - **ID:** unique identifier for each chunk.
  - **Vector:** a 1024-dimensional floating-point vector
  - **Metadata:** JSON object with:
    - `source` (string): The filename of the source PDF or "direct text input".
    - `page` (integer): The page number from which the chunk was extracted.
    - `position` (integer): chunk index 

### 4. Retrieval & Reranking

- **Retriever:**
  - **Method:** The retriever is created from the Pinecone index and uses the Maximum Marginal Relevance (MMR) search type.
  - **Parameters:**
    - `search_kwargs`: `{"k": 20}` (retrieves the 20 most relevant documents).
- **Reranker:**
  - **Provider:** [Cohere](https://cohere.com/)
  - **Model:** `rerank-english-v3.0`
  - **Parameters:**
    - `top_n`: 5 (reranks and returns the top 5 most relevant documents).

### 5. Language Model (LLM) for Generation

- **Provider:** [Groq](https://groq.com/)
- **Model:** `llama-3.1-8b-instant`

### 6. API Configurations

- The application requires API keys for Cohere, Groq, and Pinecone.
- These keys are managed through a `.env` file in the root of the project.
- Refer to the "Quick Start" section for instructions on setting up the `.env` file.

## chunking params used 
- **Chunk Size:** 1000 characters
- **Chunk Overlap:** 150 characters
- **Separators:** `["\n\n", "\n"]`

## Retriever/reranker settings

### Retriever
- **Search Type:** Maximum Marginal Relevance (MMR)
- **k (number of documents to retrieve):** 20

### Reranker
- **Model:** `rerank-english-v3.0`
- **top_n (number of documents to return):** 5

## note: Cohere/hoster reranker doesnt allow custom prompt tuning through api reference which traditionals cross encoders do. 

## Providers which i usedd

- **LLM:** [Groq](https://groq.com/) (`llama-3.1-8b-instant`)
- **Embedding:** [Cohere](https://cohere.com/) (`embed-english-v3.0`)
- **Reranker:** [Cohere](https://cohere.com/) (`rerank-english-v3.0`)
- **Vector Store:** [Pinecone](https://www.pinecone.io/)
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Core Libraries:** [LangChain](https://www.langchain.com/)

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd minirag
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  #On windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Set up your environment variables:**
   - Create a file named `.env` in the root of the project.
   - Add your API keys to the `.env` file:
     ```
     GROQ_API_KEY="your-groq-api-key"
     COHERE_API_KEY="your-cohere-api-key"
     PINECONE_API_KEY="your-pinecone-api-key"
     GEMINI_API_KEY="your-gemini-api-key"
     ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run src/streamlit_app.py
   ```
## Remark 

1. I using gemini api for chat model and emebeddings but as my free tier was about to end, the api's couldn't be used and i had to shift to cohere emebed and groq llama3.1 8b. Otherwise for text, gpt 4.0 and 5 stands best along with gemini2.5 pro.  
2. With the metadata part, I  could use deepdoctection (https://github.com/deepdoctection/deepdoctection) (lib which combines PDF mining and VLMs both customizable) to get layout data and get precise section , subsection , page number and even charts/image descriptions accuractely. I am familiar and have worked with it before. However , due to limited reference about multimodal capabilities in the task and considering upload as purely text, i proceeded with textual insights only. 
For future work, I would surely go for this option as it is much better and convinienet than using going for a VLM and pdf miner seperately. There's a balance to be maintained with this. 
Unstructuredio also provides the same thing but smartly using LLMs only however it is paid.
3. I initally made a FASTAPI + Express JS vercel which was running properly on my local system, however encountered an issue with vercel which, due to time constraint (ongoing midsemcan you ca  exams and the 72 hr deadline) had to leave after spending quite some time and so deployed it on HF spaced using streamlit as frontend and docker. 
I have a running app which is still running fine locally. 
4. For estimates, I used tiktoken to get size with rough cost estimates available on the internet , I have predicted those. 
5. Minimal eval and the text/dataset used is also attached in the repository. 

### Eval
from attached golden set and qa pairs, 
- **Precision** = 0.83  
- **Recall** = 0.95  
- **Success Rate / Accuracy** = 1.0

**Note:**  
- **Recall**  high recall here means most key information was captured.  
- **Precision**  slightly lower due to extra details like case numbers or additional cancers.  
- **Success Rate / Accuracy** : 5 correct answers out of 5 = > 1 
