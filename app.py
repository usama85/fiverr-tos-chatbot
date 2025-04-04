import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import os
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Initialize the conversational LLM
@st.cache_resource
def initialize_llm():
    try:
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),
            temperature=0.5,
            max_new_tokens=100
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.stop()

# Initialize ChromaDB
@st.cache_resource
def initialize_chroma_db():
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection(name="pdf_collection")
        return collection
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {e}")
        st.stop()

# Set up embedding model
@st.cache_resource
def setup_embedding_model():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"}
        )
        return embedding_model
    except Exception as e:
        st.error(f"Error initializing embedding model: {e}")
        st.stop()

# Query database for relevant documents
def retrieve_relevant_documents(collection, embedding_model, query, k=5, score_threshold=0.5):
    try:
        query_embedding = embedding_model.embed_query(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        relevant_docs = []
        metadata_list = []
        
        for i in range(len(results["documents"][0])):
            score = results["distances"][0][i]
            if score >= score_threshold:
                relevant_docs.append(results["documents"][0][i])
                metadata_list.append(results["metadatas"][0][i])
                
        if not relevant_docs:
            return "", []
        else:
            return "\n".join(relevant_docs), metadata_list
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return "", []

# Create prompt template
def create_prompt_template():
    template = """<s>[INST] You are a helpful AI assistant. Here are some documents that might help you answer the user's question. Keep responses according the context:
Context:
{context}

Current conversation:
{history}
Human: {input}
Assistant: [/INST]"""
    return PromptTemplate(template=template, input_variables=["context", "history", "input"])

# Format chat history for prompt
def format_history_for_prompt(history_list):
    formatted_history = ""
    for message in history_list:
        formatted_history += f"Human: {message['user']}\nAssistant: {message['bot']}\n"
    return formatted_history

# Main Streamlit app
def main():
    st.set_page_config(page_title="Fiverr TOS Chatbot", page_icon="📚")
    st.title("Fiverr TOS Chatbot")
    st.markdown("Ask questions about Fiverr's Terms of Service")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize components
    llm = initialize_llm()
    collection = initialize_chroma_db()
    embedding_model = setup_embedding_model()
    prompt = create_prompt_template()
    conversation_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(message["user"])
        with st.chat_message("assistant"):
            st.write(message["bot"])
            if "documents" in message:
                with st.expander("View Retrieved Documents"):
                    for i, (doc, metadata) in enumerate(zip(message["documents"], message["metadata"])):
                        st.markdown(f"**Document {i+1}**")
                        st.markdown(f"**Source:** {metadata.get('source', 'Unknown')}")
                        st.text_area(f"Content {i+1}", value=doc, height=150, key=f"doc_{i}_{len(st.session_state.chat_history)}")
    
    # Chat input
    user_input = st.chat_input("Ask a question about Fiverr's TOS")
    
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Process and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant documents
                context, metadata_list = retrieve_relevant_documents(collection, embedding_model, user_input)
                
                if not context:
                    response = "I don't have relevant information to answer that question."
                    documents = []
                else:
                    # Format history for prompt
                    formatted_history = format_history_for_prompt(st.session_state.chat_history)
                    
                    # Generate response
                    response = conversation_chain.run(
                        context=context, 
                        history=formatted_history, 
                        input=user_input
                    )
                    
                    documents = context.split("\n")
                
                st.write(response)
                
                # Show retrieved documents
                if documents:
                    with st.expander("View Retrieved Documents"):
                        for i, (doc, metadata) in enumerate(zip(documents, metadata_list)):
                            st.markdown(f"**Document {i+1}**")
                            st.markdown(f"**Source:** {metadata.get('source', 'Unknown')}")
                            st.text_area(f"Content {i+1}", value=doc, height=150, key=f"resp_doc_{i}")
            
            # Update chat history
            history_entry = {
                "user": user_input, 
                "bot": response,
                "documents": documents if documents else [],
                "metadata": metadata_list if metadata_list else []
            }
            st.session_state.chat_history.append(history_entry)

if __name__ == "__main__":
    main()
