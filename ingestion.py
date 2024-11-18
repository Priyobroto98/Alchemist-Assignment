from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document
import os
import tempfile
from typing import List

load_dotenv()
os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY')
EMBEDDINGS_MODEL_NAME = os.getenv('EMBEDDINGS_MODEL_NAME')



embeddings = CohereEmbeddings(
    model=EMBEDDINGS_MODEL_NAME,
    cohere_api_key=os.getenv('COHERE_API_KEY')
)

def load_document(file):
    """Load documents based on file type."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    if file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_file_path)
    elif file.name.endswith(".txt"):
        loader = Docx2txtLoader(tmp_file_path)
    else:
        st.error("Unsupported file type!")
        return []

    documents = loader.load()
    os.remove(tmp_file_path)
    return documents

def process_documents(uploaded_files,target_source_chunks,chunk_size,chunk_overlap):
    
    """Process uploaded documents and create vectorstore and retriever."""
    
    documents = []
    for file in uploaded_files:
        documents.extend(load_document(file))

    if not documents:
        st.error("No valid documents were loaded!")
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        collection_name="my_collection",
        documents=text_chunks,
        embedding=embeddings,
        persist_directory='./chroma_db'
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": target_source_chunks})
    
    return vectorstore, retriever
