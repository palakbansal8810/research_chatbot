import os
from tqdm import tqdm
from text_extracting import process_pdf
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

def process_all_pdfs_to_chroma(input_folder, persist_dir="./chroma_store"):
    all_documents = []

    for file in os.listdir(input_folder):
        if file.lower().endswith('.pdf'):
            path = os.path.join(input_folder, file)
            print(f"[📄] Processing {file}")
            try:
                docs = process_pdf(path)
                all_documents.extend(docs)
                print(f"[✅] Added {len(docs)} chunks from {file}")
            except Exception as e:
                print(f"[❌] Failed {file}: {e}")

    print(f"[🔁] Embedding {len(all_documents)} chunks...")
    
    # Use tqdm inside embed_documents to show progress manually
    texts = [doc.page_content for doc in all_documents]
    list(tqdm(texts, desc="Queuing Chunks"))  # Visual only, doesn't change logic

    # Chroma will internally call embedding_model.embed_documents()
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        persist_directory=persist_dir,
        
    )
    vectorstore.persist()
    print(f"[🎉] Stored {len(all_documents)} chunks in ChromaDB.")
    return vectorstore.as_retriever(),vectorstore