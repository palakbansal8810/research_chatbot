import os
from tabulate import tabulate
from docs_to_db import process_all_pdfs_to_chroma
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
# Set Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model_name="llama3-8b-8192")


def ask_question(query, memory, retriever, llm, top_k=8):
    docs = retriever.vectorstore.similarity_search(query, k=top_k)

    sources = []
    for doc in docs:
        source = doc.metadata.get("source", "N/A")
        page = doc.metadata.get("page", "N/A")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 100:
            snippet = snippet[:100] + "..."
        sources.append([source, snippet, f"Page {page}, Para {chunk_id}"])

    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate

    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant helping analyze legal or compliance documents.

Your response must:
1. Use the provided context to identify answers to the user's question.
2. Avoid hallucinating — only use the given content.
3. After referencing documents, provide a short theme.

Output format:
Return your analysis as a **valid JSON object** structured like this:
{{"themes": [
    {{
        "theme_name": "Concise Title for Theme 1",
        "summary": "Brief explanation of the theme",
        "supporting_documents": ["fist_pdf_name", "second_pdf_name"],
        "evidence": "Relevant quotes or summaries that illustrate this theme"
    }}],
    Dont need a single word other than the above
}}

Context:
{context}

Question:
{question}

Answer:
"""
)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    response = qa_chain.invoke({"question": query})
    return response["answer"], sources


def initialize_chatbot(input_folder):
    retriever, vectorstore = process_all_pdfs_to_chroma(
        persist_dir="D:\\Wassertsoff\\chatbot_theme_identifier\\backend\\chroma_store"
    )

    chat_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True
    )

    return memory, retriever, llm