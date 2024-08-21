import streamlit as st
from full_python_code import LogAnalyzer

# Initialize the LogAnalyzer object
log_analyzer = LogAnalyzer('processed_logs_sample.csv', sample_size=1000)

# Streamlit application
st.title("Web Traffic Log-Based Q&A System")

# User input
question = st.text_input("Do you have a question about the logs?")

# Button click
if st.button("Ask Question"):
    if question:
        result = log_analyzer.ask_question(question)
        
        if result:
            # Display the result
            st.subheader("Answer")
            st.write(result.get('result', 'No result found.'))

            st.subheader("Source Documents")
            for i, doc in enumerate(result.get('source_documents', []), 1):
                st.write(f"{i}. {doc.page_content[:200]}...")
        else:
            st.write("No result found.")
    else:
        st.write("Please enter a question.")
