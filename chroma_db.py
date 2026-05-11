import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define the directory to persist the vector store
persist_directory = "./chroma_db"

# Initialize local HuggingFace embeddings (no API key required)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Only rebuild the vectorstore if it doesn't already exist on disk.
# This prevents schema conflicts and speeds up every subsequent startup.
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print("Loading existing vectorstore from disk...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
else:
    print("Building vectorstore from PDFs (first run only)...")

    pdf_paths = [
        "data/mmm1.pdf",
        "data/mmm2.pdf",
        "data/mmm3.pdf",
        "data/mmm4.pdf",
        "data/mmm5.pdf",
    ]

    documents = []
    for path in pdf_paths:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        else:
            print(f"Warning: PDF not found — {path}")

    if not documents:
        raise FileNotFoundError(
            "No PDF documents were loaded. "
            "Make sure your PDFs are in the data/ folder."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    print(f"Vectorstore built and saved to '{persist_directory}'.")