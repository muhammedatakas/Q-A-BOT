# full_python_code.py

import pandas as pd
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import torch
import re
import logging
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings



    
class LogAnalyzer:
    def __init__(self, csv_file_path, sample_size=1000):
        # Initialization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.df = self.load_and_preprocess_logs(csv_file_path, sample_size)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = self.create_vectorstore()
        
        logging.basicConfig(level=logging.INFO)
        self.llm = OllamaLLM(
            model="llama3",
            temperature=0.7,
            top_p=0.95,
            device=self.device
        )
        self.prompt = PromptTemplate(
            template="""
            You are an expert cybersecurity analyst specializing in web traffic log analysis. 
            Your task is to provide an in-depth analysis of these logs to answer specific questions posed by users.
            {context}
            Based on the above logs, answer the following question with detailed reasoning and examples:
            Question: {question}
            """,
            input_variables=["context", "question"]
        )
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )
    
    def parse_page_content(self, page_content):
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

    def downsample_csv(self, csv_file_path, sample_size, output_file_path):
        df = pd.read_csv(csv_file_path)
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size)
        df.to_csv(output_file_path, index=False)
        return output_file_path

    def load_and_preprocess_logs(self, csv_file_path, sample_size=1000):
        downsampled_csv = self.downsample_csv(csv_file_path, sample_size, 'downsampled_logs.csv')
        loader = CSVLoader(file_path=downsampled_csv)
        documents = loader.load()

        data = []
        for doc in documents:
            parsed_data = self.parse_page_content(doc.page_content)
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

    def create_vectorstore(self):
        documents = [
            f"{row['text']} (Datetime: {row['datetime']}, User Agent: {row['user_agent']})"
            for _, row in self.df.iterrows()
        ]
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)
        return FAISS.from_documents(texts, self.embeddings)

    def ask_question(self, question):
        try:
            result = self.rag_chain({"query": question})
            return result
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            return {"result": "An error occurred.", "source_documents": []}

