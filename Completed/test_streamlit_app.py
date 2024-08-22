import unittest
from streamlit_app import log_analyzer

class TestStreamlitApp(unittest.TestCase):

    def test_log_analyzer_initialization(self):
        # Test if the log_analyzer object is initialized correctly
        self.assertIsNotNone(log_analyzer.df)
        self.assertIsNotNone(log_analyzer.embeddings)
        self.assertIsNotNone(log_analyzer.vectorstore)
        self.assertIsNotNone(log_analyzer.llm)
        self.assertIsNotNone(log_analyzer.prompt)
        self.assertIsNotNone(log_analyzer.rag_chain)

    def test_ask_question(self):
        # Test the ask_question method
        question = "What is the most common status code?"
        result = log_analyzer.ask_question(question)
        self.assertIn('result', result)
        self.assertIn('source_documents', result)

if __name__ == '__main__':
    unittest.main()