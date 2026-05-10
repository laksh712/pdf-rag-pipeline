import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile, os

st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with Any PDF")
st.caption("RAG-powered Q&A — upload a PDF and ask anything")

groq_key = st.sidebar.text_input("🔑 Groq API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file and groq_key:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        llm = ChatGroq(api_key=groq_key, model_name="llama-3.3-70b-versatile")

        prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Context: {context}
Question: {question}
""")

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        st.success(f"✅ Indexed {len(chunks)} chunks from {len(docs)} pages")
        os.unlink(tmp_path)

    question = st.text_input("Ask a question about your PDF")
    if question:
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
            st.markdown(f"**Answer:** {answer}")
else:
    st.info("👈 Add your Groq API key and upload a PDF to get started")