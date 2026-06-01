import os
import streamlit as st



st.title("AskDoc | AI Document Analyst")
st.markdown("""
#### Upload business reports, research papers, or company data and ask questions to extract insights instantly
""")

from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings

# from dotenv import load_dotenv
# load_dotenv()
# GROQ_API_KEY=os.getenv("GROQ_API_KEY")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]



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
        # Preserve the original uploaded filename for citations
        filename = getattr(uploaded_file, "name", None) or "uploaded.pdf"
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        loaded_docs = loader.load()
        # remove the temporary file immediately after loading to avoid disk buildup
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        # Replace the default 'source' (temp path) with the original filename
        for doc in loaded_docs:
            # keep page number if available in metadata
            page = doc.metadata.get("page") or doc.metadata.get("page_number")
            if page is not None:
                doc.metadata["source"] = f"{filename} (page {page})"
            else:
                doc.metadata["source"] = filename

        docs.extend(loaded_docs)

    return docs

def get_llm():
    # Create or return a per-session LLM client to avoid shared mutable state
    if "llm" not in st.session_state:
        st.session_state.llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.3-70b-versatile")
    return st.session_state.llm
 
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
        # ensure each session has its own llm instance
        llm = get_llm()
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vectors.as_retriever(search_type="mmr",search_kwargs={"k": 6, "fetch_k": 20})
        retriever_chain=create_retrieval_chain(retriever,document_chain)
        response=retriever_chain.invoke({"input":user_input})
        answer = response.get('answer', '')
        st.write(answer)

        # Build a deduplicated, ordered list of citations from retrieved context
        citations = []
        seen = set()
        for doc in response.get("context", []):
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                citations.append(src)
                seen.add(src)

        if citations:
            st.markdown("**Citations:**")
            for i, c in enumerate(citations, start=1):
                st.markdown(f"{i}. {c}")
    
    
        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.metadata.get("source", "Unknown"))
                st.write(doc.page_content)
                st.write("--------------------------------")
                
        
if st.button("Reset"):
    st.session_state.clear()