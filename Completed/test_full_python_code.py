import unittest
from full_python_code import LogAnalyzer

class TestLogAnalyzer(unittest.TestCase):

    def setUp(self):
        # Initialize the LogAnalyzer object with a sample CSV file
        self.log_analyzer = LogAnalyzer('processed_logs_sample.csv', sample_size=10)

    def test_initialization(self):
        # Test if the LogAnalyzer object is initialized correctly
        self.assertIsNotNone(self.log_analyzer.df)
        self.assertIsNotNone(self.log_analyzer.embeddings)
        self.assertIsNotNone(self.log_analyzer.vectorstore)
        self.assertIsNotNone(self.log_analyzer.llm)
        self.assertIsNotNone(self.log_analyzer.prompt)
        self.assertIsNotNone(self.log_analyzer.rag_chain)

    def test_ask_question(self):
        # Test the ask_question method
        question = "What is the most common status code?"
        result = self.log_analyzer.ask_question(question)
        self.assertIn('result', result)
        self.assertIn('source_documents', result)

if __name__ == '__main__':
    unittest.main()