import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb

# File to track processed PDFs with their last modified timestamps
PROCESSED_FILES_LOG = "processed_pdfs.json"

def load_processed_files():
    """Load dictionary of previously processed PDFs and their modification times."""
    if os.path.exists(PROCESSED_FILES_LOG):
        try:
            with open(PROCESSED_FILES_LOG, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Error: Processed files log is corrupted. Resetting log.")
            return {}
    return {}

def save_processed_files(processed_files):
    """Save dictionary of processed PDFs and their modification times."""
    try:
        with open(PROCESSED_FILES_LOG, 'w') as f:
            json.dump(processed_files, f, indent=2)
    except Exception as e:
        print(f"Error saving processed files log: {e}")

# Step 1: Define the folder containing PDFs
print("Step 1: Setting up the PDF folder...")
pdf_folder = "pdf_docs/"
os.makedirs(pdf_folder, exist_ok=True)  # Create folder if it doesn't exist
print(f"PDF folder set to: {pdf_folder}")

# Get all PDF files in the folder
print("Scanning for PDF files in the folder...")
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF files.")

# Load record of previously processed files
print("Loading record of previously processed files...")
processed_files = load_processed_files()

# Determine which files are new or modified
print("Checking for new or modified PDF files...")
files_to_process = []
for pdf_path in pdf_files:
    try:
        last_modified = os.path.getmtime(pdf_path)
        if pdf_path not in processed_files or processed_files[pdf_path] < last_modified:
            files_to_process.append(pdf_path)
            # Update the last modified time
            processed_files[pdf_path] = last_modified
    except Exception as e:
        print(f"Error checking file {pdf_path}: {e}")

if not files_to_process:
    print("No new or modified PDF documents to process.")
    exit(0)

print(f"Processing {len(files_to_process)} new or modified PDFs...")

# Step 2: Load new or modified PDFs
print("Step 2: Loading PDF documents...")
documents = []
for pdf in files_to_process:
    try:
        loader = PyPDFLoader(pdf)
        documents.extend(loader.load())
        print(f"Loaded document: {pdf}")
    except Exception as e:
        print(f"Error loading PDF {pdf}: {e}")

if not documents:
    print("No valid documents found to process.")
    exit(0)

# Step 3: Split documents into chunks
print("Step 3: Splitting documents into chunks...")
try:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")
except Exception as e:
    print(f"Error splitting documents into chunks: {e}")
    exit(1)

# Step 4: Generate embeddings using the embedding model
print("Step 4: Generating embeddings...")
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"}  # Change to "cuda" if using GPU
    )
    embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])
    print("Embeddings generated successfully.")
except Exception as e:
    print(f"Error generating embeddings: {e}")
    exit(1)

# Step 5: Initialize ChromaDB
print("Step 5: Initializing ChromaDB...")
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Saves data locally
    collection = chroma_client.get_or_create_collection(name="pdf_collection")
    print("ChromaDB initialized successfully.")
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    exit(1)

# Step 6: Add new data to ChromaDB
print("Step 6: Adding data to ChromaDB...")
try:
    # Get current count of documents in the collection to generate unique IDs
    current_count = len(collection.get(include=[])["ids"])
    for i, doc in enumerate(docs):
        doc_id = str(current_count + i)
        collection.add(
            ids=[doc_id],
            documents=[doc.page_content],
            metadatas=[{"source": doc.metadata.get("source", "unknown")}]
        )
    print(f"Added {len(docs)} new chunks to ChromaDB.")
except Exception as e:
    print(f"Error adding documents to ChromaDB: {e}")
    exit(1)

# Step 7: Save the updated processed files record
print("Step 7: Saving processed files record...")
save_processed_files(processed_files)
print("Processed files record saved successfully.")

# Final summary
print(f"Added {len(docs)} new chunks from {len(files_to_process)} PDFs to ChromaDB.")
print(f"Total documents in collection: approximately {current_count + len(docs)}")
