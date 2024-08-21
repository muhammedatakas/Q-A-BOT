# You need to download cuda if you want run the faiss with gpu
# You need to install like this if you are using windows conda forge::faiss-gpu

import pandas as pd
import numpy as np
from datetime import datetime

from tqdm import tqdm
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch
import re
import logging
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import CSVLoader

# Check for GPU availability
torch.cuda.empty_cache()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Function to parse the page content
def parse_page_content(page_content):
    pattern = re.compile(
        r'IP: (?P<ip>[\d\.]+)\n'
        r'Identity: (?P<identity>.+)\n'
        r'User: (?P<user>.+)\n'
        r'Timestamp: (?P<datetime>.+)\n'
        r'Request: (?P<method>\w+) (?P<url>.+) HTTP/\d\.\d\n'
        r'Status: (?P<status>\d+)\n'
        r'Size: (?P<size>\d+)\n'
        r'Referer: (?P<referer>.+)\n'
        r'User-Agent: (?P<user_agent>.+)'
    )
    match = pattern.search(page_content)
    if match:
        return match.groupdict()
    return {}


# Function to downsample the CSV file
def downsample_csv(csv_file_path, sample_size, output_file_path):
    df = pd.read_csv(csv_file_path)
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size)
    df.to_csv(output_file_path, index=False)
    return output_file_path


# Load and preprocess logs using CSVLoader
def load_and_preprocess_logs(csv_file_path, sample_size=10000):
    downsampled_csv = downsample_csv(csv_file_path, sample_size, 'downsampled_logs.csv')
    loader = CSVLoader(file_path=downsampled_csv)
    documents = loader.load()

    data = []
    for doc in documents:
        parsed_data = parse_page_content(doc.page_content)
        data.append(parsed_data)
    df = pd.DataFrame(data)

    df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%b/%Y:%H:%M:%S %z', errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday
    df['status'] = df['status'].astype(int, errors='ignore')
    df['size'] = df['size'].astype(int, errors='ignore')
    df['status_category'] = df['status'] // 100
    df['text'] = df.apply(
        lambda row: f"{row['method']} {row['url']} (Status: {row['status']}, Size: {row['size']}, IP: {row['ip']})",
        axis=1)

    return df


# Example usage
processed_logs = load_and_preprocess_logs('processed_logs_sample.csv', sample_size=1000)

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create documents for vector store
documents = [
    f"{row['text']} (Datetime: {row['datetime']}, User Agent: {row['user_agent']})"
    for _, row in processed_logs.iterrows()
]

# Split documents
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.create_documents(documents)

# Create the vector store
vectorstore = FAISS.from_documents(texts, embeddings)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize OllamaLLM
llm = OllamaLLM(
    model="llama3",
    temperature=0.7,
    top_p=0.95,
)

template = """
    You are an expert cybersecurity analyst specializing in web traffic log analysis. You have access to detailed log entries from a high-traffic website. Your task is to provide an in-depth analysis of these logs to answer specific questions posed by users.

    When analyzing the logs, follow these guidelines:
    1. Data Integrity: Ensure the accuracy and completeness of the data by cross-referencing multiple log entries where applicable.
    2. Pattern Recognition: Identify and explain any significant patterns or anomalies in user behavior, such as:
       - Repeated access from specific IP addresses.
       - Unusual patterns in user-agent strings (indicating bots, crawlers, or potential attackers).
       - Consistent access to specific pages or endpoints at unusual times.
    3. Contextual Correlation: Correlate log entries across different dimensions (e.g., IP, time, URL) to build a coherent narrative of the events.
    4. Security Implications: Assess and highlight any potential security concerns, such as:
       - Signs of DDoS attacks.
       - Unusual traffic spikes that could indicate brute force attempts.
       - Access patterns that suggest vulnerability scanning.
    5. Detailed Justification: For each observation or conclusion, provide specific log entries as evidence. Explain how these logs lead to your conclusions.

    If the information requested is not available in the logs or if you are unable to determine an answer, clearly state that the data is inconclusive.

    Below are the relevant log entries:

    {context}

    Based on the above logs, answer the following question with detailed reasoning and examples:

    Question: {question}
    """

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)


# Function to ask a question
def ask_question(question):
    result = rag_chain({"query": question})
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print("\nSource Documents:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"{i}. {doc.page_content[:200]}...")


# Example questions
ask_question(
    "Are there any unusual patterns in user-agent strings that might indicate bot activity or potential attackers?")
ask_question(
    "Which HTTP methods are predominantly used in the logs, and what does this tell us about the nature of the traffic?")