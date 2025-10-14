import langchain 
from langchain.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time 
import os 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings

from langchain_community.docstore.document import Document
from .reranker import reranker_retriever as reranker
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain as CRC
from .llm import llm, prompt
#from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def load_and_chunk_pdf(pdf_file_path):
    """Load and chunk a PDF file with metadata"""
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load()
    
    for doc in pages:
        doc.metadata['title'] = doc.metadata.get('title', None)
        doc.metadata['source'] = os.path.basename(pdf_file_path)
        doc.metadata['page'] = doc.metadata['page']+1

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", '\n']
    )

    chunks = text_splitter.split_documents(pages)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata['position'] = i
        chunk.metadata['total_chunks'] = len(chunks)

    return chunks


def chunk_text(text, source= "direct text input "):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", '\n']
    )
    
    docs = [Document(page_content=text, metadata={"source": source})]
    chunks = text_splitter.split_documents(docs)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata['position'] = i
        chunk.metadata['total_chunks'] = len(chunks)

    return chunks


def store_embeddings(chunks):
    embeddings = CohereEmbeddings(model="embed-english-v3.0")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'miniragindex'
    
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    for chunk in chunks:
        if hasattr(chunk, 'metadata'):
            if 'title' not in chunk.metadata or chunk.metadata['title'] is None:
                chunk.metadata['title'] = ''
            
            chunk.metadata = {
                k: ('' if v is None else str(v) if isinstance(v, (int, float, bool)) else v)
                for k, v in chunk.metadata.items()
            }
    
    vector_store = PineconeVectorStore(
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=PINECONE_API_KEY
    )

    vector_store.add_documents(chunks)
    return vector_store

def format_docs_with_metadata(docs):
    formatted = []
    
    for i, doc in enumerate(docs):
        formatted.append(
            f"[Chunk {i}]\n"
            f"Content: {doc.page_content}\n"
            f"---"
        )
    
    return "\n\n".join(formatted)

def create_retrieval_chain_final(llm, vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 20}
    )
    compression_retriever = reranker(retriever)
    
    citation_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_prompt=None,
        document_separator="\n\n"
    )
    
    retrieval_chain = CRC(
        retriever=compression_retriever,
        combine_docs_chain=citation_chain
    )
    
    return retrieval_chain