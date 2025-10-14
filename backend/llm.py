from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=2048,
    groq_api_key=GROQ_API_KEY
)


prompt = ChatPromptTemplate.from_template(
"""You are a research assistant. Answer the question using ONLY the provided sources.

CRITICAL RULES:
1. Each source below is numbered [Chunk 0], [Chunk 1], [Chunk 2], etc.
2. When you make a claim, cite it using the chunk number like this: "The data shows X values. [0]" or "According to the research, it is stilll under development. [1][2]". Add the markes ONLY after sentence completion.
3. Use inline citations throughout your answer - put the chunk number in square brackets after each claim
4. If sources don't answer the question, say "I cannot answer based on the provided sources"
5. NEVER make up information
6. ANSWER IN NATURAL LANGUAGE ALONG WITH CITATION MARKERS AS STATED ABOVE.
7. DISPLAY ONLY THE CHUNKS MARKERS YOU USED AS CITATIONS IN THE ANSWER.
8. DO NOT REPEAT THE SAME INFORMATION STRICTLY IN THE ANSWER.
9. IF SUFFICIENT INFORMATION IS THERE IN ONE CHUNK , DO NOT CITE OTHER CHUNKS FOR THE SAME INFORMATION. YOU MAY USE THOSE FOR ANSWERING BUT NOT FOR CITATION.
Sources:
{context}

Question: {input}

Answer with inline chunk number citations [0], [1], [2], etc.
DO NOT MENTION 'Used Chunks : <chunk numbers> ' IN THE END OF THE ANSWER.
"""
)

def get_citation_prompt():
    """Return the citation prompt"""
    return prompt