from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import os
import warnings
import json
from langchain_core._api.deprecation import LangChainDeprecationWarning

# Suppress only LangChain deprecation warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
# Suppresses ALL warnings (not recommended in production)
warnings.filterwarnings("ignore")

# Load environment variables from a .env file
load_dotenv()

# Step 1: Initialize the conversational LLM
def initialize_llm():
    """Initialize the conversational LLM."""
    print("Initializing the conversational LLM...")
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # Model repository ID
            huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),  # API token from environment
            temperature=0.5,  # Controls randomness in responses
            max_new_tokens=100  # Maximum number of tokens to generate
        )
        print("Conversational LLM initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        exit(1)

# Step 2: Initialize ChromaDB for RAG
def initialize_chroma_db():
    """Initialize ChromaDB and load the collection."""
    print("Initializing ChromaDB for RAG...")
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Path to the local ChromaDB
        collection = chroma_client.get_or_create_collection(name="pdf_collection")
        print("ChromaDB initialized successfully.")
        return collection
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        exit(1)

# Step 3: Set up the embedding model
def setup_embedding_model():
    """Set up the embedding model."""
    print("Setting up the embedding model...")
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

# Step 4: Query the vector database for relevant documents
def retrieve_relevant_documents(collection, embedding_model, query, k=5, score_threshold=0.5):
    """
    Retrieve relevant documents from the vector database based on the query.

    Args:
        collection: The ChromaDB collection object.
        embedding_model: The embedding model used for the query.
        query: The query string.
        k: Number of top results to retrieve.
        score_threshold: Minimum similarity score to consider a result relevant.

    Returns:
        Tuple containing:
        - A single concatenated string of relevant documents.
        - A list of metadata for the relevant documents.
    """
    print("Retrieving relevant documents from the vector database...")
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
        relevant_docs = []
        metadata_list = []
        for i in range(len(results["documents"][0])):  # Iterate over the first (and only) query's results
            score = results["distances"][0][i]
            if score >= score_threshold:
                relevant_docs.append(results["documents"][0][i])
                metadata_list.append(results["metadatas"][0][i])

        if not relevant_docs:
            print("No relevant documents found above the score threshold.")
            return "", []
        else:
            print(f"Found {len(relevant_docs)} relevant documents.")
            return "\n".join(relevant_docs), metadata_list
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return "", []

# Step 5: Define the conversation prompt template
def create_prompt_template():
    """Create the prompt template for the conversational LLM."""
    print("Creating the prompt template...")
    template = """<s>[INST] You are a helpful AI assistant. Here are some documents that might help you answer the user's question. Keep responses according the context:
Context:
{context}

Current conversation:
{history}
Human: {input}
Assistant: [/INST]"""
    print("Prompt template created successfully.")
    return PromptTemplate(template=template, input_variables=["context", "history", "input"])

# Step 6: Save and load chat history
def save_chat_history(history, filename="chat_history.json"):
    """Save chat history to a JSON file."""
    with open(filename, "w") as f:
        json.dump(history, f, indent=2)

def load_chat_history(filename="chat_history.json"):
    """Load chat history from a JSON file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Step 7: Format chat history for the prompt
def format_history_for_prompt(history_list):
    """Convert history list to string format for the prompt."""
    formatted_history = ""
    for message in history_list:
        formatted_history += f"Human: {message['user']}\nAssistant: {message['bot']}\n"
    return formatted_history

# Main function
if __name__ == "__main__":
    print("Starting the RAG conversational chat model...")

    # Initialize components
    llm = initialize_llm()
    collection = initialize_chroma_db()
    embedding_model = setup_embedding_model()
    prompt = create_prompt_template()

    # Create a conversational chain
    conversation_chain = LLMChain(llm=llm, prompt=prompt)

    # Load chat history
    chat_history = load_chat_history()

    # Print initial greeting
    print("\nAI: Hello! How can I help you today? (Type 'exit' to end)\n")

    # Start the conversation loop
    while True:
        # Get user input
        user_input = input("You: ")

        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("AI: Goodbye!")
            save_chat_history(chat_history)  # Save history before exiting
            break

        # Retrieve relevant documents from the vector database
        context, metadata_list = retrieve_relevant_documents(collection, embedding_model, user_input)

        # If no relevant documents are found, respond with "I do not know."
        if not context:
            response = "I do not know."
        else:
            # Format history for the prompt
            formatted_history = format_history_for_prompt(chat_history)

            # Generate AI response using the conversation chain
            response = conversation_chain.run(context=context, history=formatted_history, input=user_input)

            # Append the exact retrieved document chunks and their sources to the response
            for i, (chunk, metadata) in enumerate(zip(context.split("\n"), metadata_list)):
                response += f"\n\nRelevant Information {i + 1}:\n{chunk}\nSource: {metadata.get('source', 'Unknown')}"

        # Update conversation history
        chat_history.append({"user": user_input, "bot": response})

        # Save chat history after each interaction
        save_chat_history(chat_history)

        # Print the AI's response
        print(f"\nAI: {response}\n")