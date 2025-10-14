
import os 
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
def reranker_retriever(retriever):
    compressor = CohereRerank(model="rerank-english-v3.0", top_n=5, cohere_api_key=COHERE_API_KEY)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever

