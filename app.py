from chain import *
from ingestion import *
import streamlit as st

target_source_chunks = 2
chunk_size = 500
chunk_overlap = 50

def initialize_session_state():
    """Initialize Streamlit session state."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []



# Main Execution
def main():
        
    
    st.title("Document Q&A Chatbot")
    st.sidebar.header("Upload and Process Documents")
    initialize_session_state()

    uploaded_files = st.sidebar.file_uploader(
        "Upload your documents (PDF or TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        vectorstore, retriever = process_documents(uploaded_files,target_source_chunks,chunk_size,chunk_overlap)
        if retriever:
            handle_chat(llm, retriever)

if __name__ == "__main__":
    main()
