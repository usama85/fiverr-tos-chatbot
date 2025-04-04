import os
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings

# Step 1: Initialize ChromaDB and load the collection
def initialize_chroma_db():
    """Initialize ChromaDB and load the collection."""
    print("Step 1: Initializing ChromaDB...")
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Path to the local ChromaDB
        collection = chroma_client.get_or_create_collection(name="pdf_collection")
        print("ChromaDB initialized successfully.")
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        exit(1)

# Step 2: Set up the embedding model
def setup_embedding_model():
    """Set up the embedding model."""
    print("Step 2: Setting up the embedding model...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"}  # Change to "cuda" if using GPU
        )
        print("Embedding model initialized successfully.")
        return embedding_model
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        exit(1)

# Step 3: Query the database
def query_database(collection, embedding_model, query, k=3, score_threshold=0.6):
    """
    Query the ChromaDB collection for relevant documents based on the query.

    Args:
        collection: The ChromaDB collection object.
        embedding_model: The embedding model used for the query.
        query: The query string.
        k: Number of top results to retrieve.
        score_threshold: Minimum similarity score to consider a result relevant.

    Returns:
        List of relevant documents with metadata and scores.
    """
    print("Step 3: Querying the database...")
    try:
        # Generate embeddings for the query
        query_embedding = embedding_model.embed_query(query)
        print("Query embedding generated successfully.")

        # Perform the search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        print("Query executed successfully.")

        # Filter results based on the score threshold
        filtered_results = []
        for i in range(len(results["documents"][0])):  # Iterate over the first (and only) query's results
            score = results["distances"][0][i]  # Access the inner list of distances
            if score >= score_threshold:
                filtered_results.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": score
                })

        if not filtered_results:
            print("No relevant documents found above the score threshold.")
        else:
            print(f"Found {len(filtered_results)} relevant documents.")
        return filtered_results
    except Exception as e:
        print(f"Error querying the database: {e}")
        exit(1)

# Step 4: Display the results
def display_results(results):
    """Display the retrieved results."""
    print("Step 4: Displaying results...")
    if not results:
        print("No results to display.")
        return

    for i, result in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(f"Document: {result['document']}")
        print(f"Metadata: {result['metadata']}")
        print(f"Score: {result['score']:.2f}")

# Main function
if __name__ == "__main__":
    print("Starting the retrieval process...")

    # Step 1: Initialize ChromaDB
    collection = initialize_chroma_db()

    # Step 2: Set up the embedding model
    embedding_model = setup_embedding_model()

    # Step 3: Define the query
    query = "what are the payment methods are allowed in fiverr?."  # Hardcoded query
    print(f"Query: {query}")

    # Step 4: Query the database
    results = query_database(collection, embedding_model, query, k=5, score_threshold=0.5)

    # Step 5: Display the results
    display_results(results)

    print("Retrieval process completed.")