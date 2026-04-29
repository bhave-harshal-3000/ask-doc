import os
import streamlit as st

st.title("Document QnA")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")



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

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFDirectoryLoader("documents")                                                   # data injestion
        st.session_state.docs=st.session_state.loader.load()                                                        # document loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)           # splitter initiated
        st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs)      # chunking and splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)      # vector store created
 
      
if st.button("Document Embedding"):
    vector_embedding()
    st.write("vector db is ready...") 
 
input = st.text_input("Enter your questions: ")


if input:
    if "vectors" in st.session_state:
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever()
        retriever_chain=create_retrieval_chain(retriever,document_chain)
        response=retriever_chain.invoke({"input":input})
        st.write(response['answer'])
    
    
        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
                
    else:
        st.error("Please click 'Document Embedding' first to initialize the database.")