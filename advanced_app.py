import os
import streamlit as st



st.title("AskDoc | AI Document Analyst")
st.markdown("""
#### Upload business reports, research papers, or company data and ask questions to extract insights instantly
""")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")




uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


from langchain_community.document_loaders import PyPDFLoader
import tempfile

def process_uploaded_files(uploaded_files):
    docs = []

    for uploaded_file in uploaded_files:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs.extend(loader.load())

    return docs

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)
 
prompt = ChatPromptTemplate.from_template(
    """Answer the following questions based on the provided context only
    Please provided the most accurate response based on the question
    
    <context>
    {context}
    <context>
    
    Questions : {input}
    
    """
)

def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:

        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.docs = process_uploaded_files(uploaded_files)
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)
        
       
if st.button("Create Vectorstore"):
    if uploaded_files:
        with st.spinner("Analyzing documents... Please hold on!"):
            vector_embedding(uploaded_files)
        st.success("Documents processed successfully!")
    else:
        st.error("Please upload at least one PDF.")
 
user_input = None
if "vectors" in st.session_state:
    user_input = st.text_input("Ask Questions about your data:")


if user_input:
    if "vectors" in st.session_state:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever(search_type="mmr",search_kwargs={"k": 6, "fetch_k": 20})
        retriever_chain=create_retrieval_chain(retriever,document_chain)
        response=retriever_chain.invoke({"input":user_input})
        st.write(response['answer'])
    
    
        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.metadata.get("source", "Unknown"))
                st.write(doc.page_content)
                st.write("--------------------------------")
                
        
if st.button("Reset"):
    st.session_state.clear()