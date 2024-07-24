import json
import sys
import os
import boto3

# we will be using titan embeddings model to create embeddings from langchain library
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock as Bedrock

# For data ingestion
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For Vector Store
from langchain_community.vectorstores import FAISS

#For Prompt Template
from langchain_core.prompts import PromptTemplate

# For Retrieval Augmented Generation
from langchain.chains import RetrievalQA

# Initializing Bedrock Runtime client
bedrock_rt = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

# Initializing Bedrock Embeddings
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",  # "cohere.embed-english-v3", #amazon.titan-embed-text-v1
    client=bedrock_rt,
    region_name="us-east-1"
    )

# Data Ingestion
def data_ingestion():
    loader = PyPDFDirectoryLoader("Data")
    documents = loader.load()
    print(f"Number of documents loaded: {len(documents)}")

# Splitting data into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000,
        chunk_overlap = 1000
    )
    doc = text_splitter.split_documents(documents)
    print(f"Number of chunks after splitting: {len(doc)}")
    return doc

# Creating Vector Store
def vector_store(doc):
    try:
        print(f"Number of document chunks: {len(doc)}")
        if len(doc) == 0:
            print("No documents to process. Check your data ingestion.")
            return

        print("Generating embeddings...")
        vector_store_faiss = FAISS.from_documents(
            doc,
            bedrock_embeddings
        )
        print("Embeddings generated successfully.")
        vector_store_faiss.save_local("faiss_index")
        print("Vector store saved locally.")
    except Exception as e:
        print(f"Error in vector_store function: {e}")
        raise

# Create LLM model
def get_claude_llm():
    llm = Bedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0", #"anthropic.claude-instant-v1",
        client=bedrock_rt,
        model_kwargs = {
            "max_tokens" : 512
        }
    )
    return llm

def get_llam3_llm():
    llm = Bedrock(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock_rt,
        model_kwargs = {
            "max_gen_len" : 512
        }
    )
    return llm

# Create PromptTemplate
promtTemplate = """
    Human : Use the following pieces of context to provide a concise answer to the question at the end but atleast summarize with 250 words with details explanation. If you don't know the answer, just say you don't know

    <context>
    {context}
    </context>

    Question : {question}
    Assistant :
"""
prompt = PromptTemplate(
    template = promtTemplate,
    input_variables = ["context", "question"]
)

# Get response from the model
def get_response_llm(llm, vector_store_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vector_store_faiss.as_retriever(search_type = "similarity", search_kwargs = {"k": 3}),
        return_source_documents = True,
        chain_type_kwargs = {"prompt": prompt}
    )
    answer = qa({"query": query})
    return answer["result"]

import streamlit as st

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock")
    query = st.text_input("Ask a question from the PDF files")

    with st.sidebar:
        st.title("Update or create vector store")

        if st.button("Vector update"):
            with st.spinner("Processing..."):
                doc = data_ingestion()
                vector_store(doc)
                st.success("Done")
   
    if st.button("Clude Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_claude_llm()
            st.write(get_response_llm(llm, faiss_index, query))
            st.success("Done")
    
    if st.button("Llama2 Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            llm = get_llam3_llm()
            st.write(get_response_llm(llm, faiss_index, query))
            st.success("Done")
if __name__ == "__main__":
    main()
    