import streamlit as st
import tempfile
import os
import shutil
import pandas as pd
from query_retriever import ask_question
from docs_to_db import process_all_pdfs_to_chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
##
from dotenv import load_dotenv
load_dotenv()
###
# Page config

st.set_page_config(page_title="Research Doc Chatbot", layout="wide")
st.title("📄 Document Research & Theme Identification Chatbot")

# Set Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# temp folder
if "upload_dir" not in st.session_state:
    st.session_state.upload_dir = tempfile.mkdtemp()

# upload PDFs 
if "retriever" not in st.session_state:
    uploaded_files = st.file_uploader("📤 Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(st.session_state.upload_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

        with st.spinner("🔄 Processing documents..."):
            retriever, vectorstore = process_all_pdfs_to_chroma(
                input_folder=st.session_state.upload_dir,
                persist_dir=os.path.join(st.session_state.upload_dir, "chroma_store")
            )

            chat_history = ChatMessageHistory()
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                chat_memory=chat_history,
                return_messages=True
            )
            llm = ChatGroq(model_name="llama3-8b-8192")

            # Save in session
            st.session_state.retriever = retriever
            st.session_state.memory = memory
            st.session_state.llm = llm
            st.session_state.chat_history = []  

        st.success("✅ Documents indexed. You can start chatting now!")

# chat interface 

if "retriever" in st.session_state:
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    user_input = st.chat_input("Ask a question based on the documents...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                answer, sources = ask_question(
                    query=user_input,
                    memory=st.session_state.memory,
                    retriever=st.session_state.retriever,
                    llm=st.session_state.llm
                )
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                if sources:
                    df = pd.DataFrame(sources, columns=["PDF File", "Extracted Answer", "Citation"])
                    st.table(df)

                st.markdown(answer)

if "first_load" not in st.session_state:
    st.session_state.clear()
    st.session_state.first_load = True
