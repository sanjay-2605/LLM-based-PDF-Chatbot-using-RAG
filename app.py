import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

st.title("LLM-based PDF Chatbot using RAG")

pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    query = st.text_input("Ask Questions From PDF")

    if query:
        docs = vectorstore.similarity_search(query)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")

        response = chain.run(input_documents=docs, question=query)

        st.write(response)
