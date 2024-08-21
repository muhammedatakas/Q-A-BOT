# **Web Traffic Log-Based Q&A System**

## **Overview**
This project is a web traffic log-based question-answering system developed using a Retrieval-Augmented Generation (RAG) model. It processes web traffic logs, stores them in a vector database, and allows users to query the logs using natural language. The system leverages two language models: LLaMA 3 and Google T5, to generate accurate and contextually relevant answers.

## **Features**
- **Log Parsing:** Efficiently parses and preprocesses web traffic logs to extract essential information.
- **Vector Database:** Utilizes FAISS for fast and efficient storage and retrieval of log data.
- **RAG Model:** Combines retrieval from the vector database with natural language generation to answer user queries.
- **GPU Optimization:** Automatically detects and utilizes GPU for faster processing if available.

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/muhammedatakas/Q-A-BOT.git
cd traffic-log-qa
```

### **2. Install Dependencies**
Use the following command to create and activate a Conda environment with all required packages:
```bash
conda env create -f environment.yml
conda activate my_project
```

### **3. Prepare the Data**
Ensure your web traffic logs are in CSV format and named [`processed_logs.csv`]. Place this file in the project root directory.

### **4. Running the System**
You can run the project directly in a Jupyter notebook or any Python environment that supports notebook-style execution. Here are the steps:

1. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the Notebook:**
   Open the `.ipynb` file containing the main code. Execute the cells step-by-step.

3. **Ask Questions:**
   After the system is initialized, you can query it by using the [`ask_question()`] For example:
   ```python
   ask_question("Which IP addresses are associated with a high number of failed login attempts?")
   ```

### **5. Streamlit Interface (Optional)**
If you want to create a Streamlit interface for your Q&A system, follow these additional steps:

1. **Install Streamlit:**
   ```bash
   pip install streamlit
   ```

2. **Run Streamlit App:**
   Create a [`app.py`] file with the Streamlit code (provided above), then run:
   ```bash
   streamlit run streamlit_app.py
   ```

## **Project Structure**
```
├── README.md               # Project documentation
├── requirements.txt        # Required packages
├── environment.yml         # Conda environment configuration
├── traffic-log-qa.ipynb    # Main Jupyter notebook
├── processed_logs.csv      # Web traffic log data
├── streamlit_app.py                  # Streamlit app (optional)
```

## **Usage**
- **Data Processing:** The system processes and structures your web traffic logs for efficient retrieval and analysis.
- **Q&A System:** Users can interact with the system by asking questions related to the logs, and the system will provide detailed answers supported by relevant log entries.
- **Streamlit Interface:** Provides a user-friendly web interface for interacting with the Q&A system.

## **Customization**
- **Model Selection:** The system currently uses LLaMA 3 and Google T5. You can replace these with other models as per your requirements.
- **Data Preprocessing:** Adjust the log parsing and preprocessing functions to suit the structure of your web traffic logs.

## **Future Enhancements**
- **Advanced Analytics:** Incorporate more sophisticated analytics to identify patterns and trends in the logs.
- **Real-time Processing:** Extend the system to handle real-time log data and provide immediate insights.
- **User Authentication:** Implement user roles and permissions for accessing sensitive log data.

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request with your proposed changes.

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

## **Contact**
For any questions or issues, feel free to open an issue in the repository or contact the project maintainer.