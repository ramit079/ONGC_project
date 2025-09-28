from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from io import StringIO
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from streamlit2 import uploadname
# import google.generativeai as genai
# client = genai.Client(api_key="AIzaSyDjuSvypHHBC1KbLGfusdQgRIp7CdseCYA")
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings

# def get_document(path):
#     loader_pdf = PyPDFLoader(path)
#     global pages
#     pages = loader_pdf.load()

# path = ""
# loader_pdf = PyPDFLoader(path)
# pages = loader_pdf.load()
# for i in pages:
#     i.page_content = ' '.join(i.page_content.split())
# char_splitter = CharacterTextSplitter(
# separator = ".",
# chunk_size = 500,
# chunk_overlap  = 50
# )
# pages_split = char_splitter.split_documents(pages)
# embedding = OllamaEmbeddings(model = "mxbai-embed-large:latest")
# vectorstore = Chroma.from_documents(documents = pages_split, 
#             embedding = embedding, 
#             persist_directory = "./vectorbase")

# embedding = OllamaEmbeddings(model = "mxbai-embed-large:latest")
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# embedding = SentenceTransformer("dunzhang/stella_en_1.5B_v5")
# embedding = genai.GenerativeModel("gemini-embedding-exp-03-07")
embedding = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = Chroma(persist_directory = f'./{uploadname}',
                embedding_function = embedding)
retriever = vector.as_retriever()
chat = OllamaLLM(   
    model='llama3:latest',
    temperature=0.7
)

# chat = OllamaLLM(
#     base_url='http://192.168.1.50:11434',
#     model='mistral-nemo',
#     temperature=0.7
# )

TEMPLATE = '''
Answer the following question:
{question}

To answer the question, use only the following context:
{context}
'''
chat_template = PromptTemplate.from_template(TEMPLATE)

chain = ({'context' : retriever,
        'question' : RunnablePassthrough()} | chat_template | chat | StrOutputParser())
    


