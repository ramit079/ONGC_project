import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import finalwithdef
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
# import google.generativeai as genai
# client = genai.Client(api_key="AIzaSyDjuSvypHHBC1KbLGfusdQgRIp7CdseCYA")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
import shutil


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes):
    """Cache PDF processing to prevent reprocessing"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        for doc in documents:
            doc.page_content = ' '.join(doc.page_content.split())
        
        char_splitter = CharacterTextSplitter(
            separator=".",
            chunk_size=500,
            chunk_overlap=50
        )
        pages_split = char_splitter.split_documents(documents)
        
        # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        #embedding = SentenceTransformer('all-MiniLM-L6-v2')
        st.write("Embedding now")
        embedding = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        file_count=0
        for item in os.listdir('allfiles'):
            item_path = os.path.join('allfiles', item)
            if os.path.isfile(item_path):
                file_count += 1

        st.write(f"Files in allfiles: {file_count}")
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embedding,
            persist_directory=f"./{uploadname}"
        )
        return vectorstore
    finally:
        os.unlink(tmp_file_path)

def get_llm_response(question):
    """Get response using cached vectorstore"""
    if not st.session_state.vectorstore:
        return "Please upload a PDF document first!"
    
    # Your actual LLM chain logic here
    # Replace with: finalwithdef.chain.invoke(question)
    return finalwithdef.chain.invoke(question)

# Application header
st.title("PDF Chatbot with Custom LLM")
st.markdown("Upload a PDF document and chat with its contents")

# PDF upload and processing
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
uploadname = uploaded_file.name

if uploaded_file and uploaded_file != st.session_state.processed_file:
    with st.spinner("Processing PDF..."):
        try:
            st.session_state.vectorstore = process_pdf(uploaded_file.getvalue())
            st.session_state.processed_file = uploaded_file
            st.success("PDF processed successfully!")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.session_state.vectorstore = None

# Single Chat interface header - ONLY ONCE
st.divider()
st.subheader("Chat with Document")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process user input
if prompt := st.chat_input("Ask about the PDF"):
    if not st.session_state.vectorstore:
        st.warning("Please upload a PDF document first!")
        st.stop()
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate LLM response with proper container management
    with st.chat_message("assistant"):
        response_container = st.empty()  # Key fix: use empty container
        
        with response_container:
            with st.spinner("Thinking..."):
                try:
                    response = get_llm_response(prompt)
                except Exception as e:
                    response = f"Error generating response: {str(e)}"
        
        # Replace spinner with actual response
        response_container.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update the interface
    st.rerun()
