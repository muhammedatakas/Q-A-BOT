
import streamlit as st
import torch

# RAG model and other necessary modules
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_ollama import OllamaLLM
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from rag_system_code import ask_question

# Streamlit title
st.title("Web Traffic Log-Based Q&A System")

# Get question from the user
question = st.text_input("Enter your question here:")

# Process the question and show results
if st.button("Ask the Question"):
    if question:
        # Get answer from RAG model
        result = ask_question(question)
        
        # Display the answer and source documents
        st.write(f"**Question:** {question}")
        st.write(f"**Answer:** {result['result']}")
        
        st.write("**Source Documents:**")
        for i, doc in enumerate(result['source_documents'], 1):
            st.write(f"{i}. {doc.page_content[:200]}...")
    else:
        st.warning("Please enter a question.")
