import streamlit as st
import os
import time
import re
from dotenv import load_dotenv
import tiktoken

from backend.main import load_and_chunk_pdf, chunk_text, store_embeddings, create_retrieval_chain_final
from backend.llm import llm
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        if response.llm_output and 'token_usage' in response.llm_output:
            self.input_tokens += response.llm_output['token_usage']['prompt_tokens']
            self.output_tokens += response.llm_output['token_usage']['completion_tokens']

st.set_page_config(
    page_title="Mini rag",
    layout="wide",
)

st.markdown("""
<style>
    .stApp {
        background-color: #0008;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #0008;
    }
</style>
""", unsafe_allow_html=True)

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

def calculate_cost(input_tokens, output_tokens, reranker_requests):
    llm_input_cost = (input_tokens / 1_000_000) * 0.05
    llm_output_cost = (output_tokens / 1_000_000) * 0.08

    reranker_cost = reranker_requests * (2.00 / 1000)

    total_cost = llm_input_cost + llm_output_cost + reranker_cost
    return total_cost

def calculate_embedding_cost(chunks):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(tokenizer.encode(chunk.page_content)) for chunk in chunks)
    embedding_cost = (total_tokens / 1_000_000) * 0.10
    return embedding_cost

def extract_citation_numbers(answer_text):
    pattern = r'\[(\d+)\]'
    citations = re.findall(pattern, answer_text)
    seen = set()
    unique_citations = []
    for c in citations:
        num = int(c)
        if num not in seen:
            seen.add(num)
            unique_citations.append(num)
    return unique_citations

def format_response(result):
    answer_text = result.get("answer", "")
    context_docs = result.get("context", [])
    
    cited_chunk_numbers = extract_citation_numbers(answer_text)
    
    citations = []
    for chunk_num in cited_chunk_numbers:
        if chunk_num < len(context_docs):
            doc = context_docs[chunk_num]
            citations.append({
                "number": chunk_num,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
    
    return {
        "answer": answer_text,
        "citations": citations
    }

with st.sidebar:
    st.header("input")
    
    uploaded_file = st.file_uploader("upload pdf", type="pdf")
    if uploaded_file:
        with st.spinner("processing pdf"):
            start_time = time.time()
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            chunks = load_and_chunk_pdf(uploaded_file.name)
            embedding_cost = calculate_embedding_cost(chunks)
            st.session_state.total_cost += embedding_cost
            
            st.session_state.vector_store = store_embeddings(chunks)
            st.session_state.retrieval_chain = create_retrieval_chain_final(llm, st.session_state.vector_store)
            
            os.remove(uploaded_file.name)
            end_time = time.time()
            st.success(f"processing time= {end_time - start_time:.2f}")
            st.info(f"emebedding cost= ${embedding_cost:.6f}")

    st.markdown("---")
    text_input = st.text_area(" or can enter text")
    if st.button("process the text ") and text_input:
        with st.spinner("processing text "):
            start_time = time.time()
            chunks = chunk_text(text_input)
            embedding_cost = calculate_embedding_cost(chunks)
            st.session_state.total_cost += embedding_cost

            st.session_state.vector_store = store_embeddings(chunks)
            st.session_state.retrieval_chain = create_retrieval_chain_final(llm, st.session_state.vector_store)
            end_time = time.time()
            st.success(f"processing time= {end_time - start_time:.2f}")
            st.info(f"emebedding cost= ${embedding_cost:.6f}")

    st.markdown("---")
    st.header("final cost")
    st.write(f"${st.session_state.total_cost:.6f}")


st.title("Mini rag")
st.markdown("A mini rag app.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "citations" in message and message["citations"]:
            with st.expander("Sources"):
                for cit in message["citations"]:
                    st.markdown(f'**[{cit["number"]}] {cit["source"]}, Page {cit["page"]}**')
                    st.markdown(f'> {cit["excerpt"]}')
        if "cost" in message:
            st.markdown(f"*Response Time: {message['response_time']:.2f}s | Cost: ${message['cost']:.6f}*")


if prompt_input := st.chat_input("ask a question"):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    if st.session_state.retrieval_chain is None:
        st.warning(" upload a PDF or enter text to begin.")
    else:
        with st.spinner("wait lemme see <thinking..>"):
            start_time = time.time()
            token_usage_callback = TokenUsageCallback()
            response = st.session_state.retrieval_chain.invoke(
                {"input": prompt_input},
                config={"callbacks": [token_usage_callback]}
            )
            end_time = time.time()
            
            formatted_response = format_response(response)
            
            input_tokens = token_usage_callback.input_tokens
            output_tokens = token_usage_callback.output_tokens
            cost = calculate_cost(input_tokens, output_tokens, 1)
            st.session_state.total_cost += cost
            
            assistant_message = {
                "role": "assistant",
                "content": formatted_response["answer"],
                "citations": formatted_response["citations"],
                "response_time": end_time - start_time,
                "cost": cost
            }
            st.session_state.messages.append(assistant_message)

            with st.chat_message("assistant"):
                st.markdown(formatted_response["answer"])
                if formatted_response["citations"]:
                    with st.expander("Sources"):
                        for cit in formatted_response["citations"]:
                            st.markdown(f'**[{cit["number"]}] {cit["source"]}, Page {cit["page"]}**')
                            st.markdown(f'> {cit["excerpt"]}')
                st.markdown(f"*Response Time: {end_time - start_time:.2f}s | Cost: ${cost:.6f}*")