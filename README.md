# LangChain RAG Chatbot

A conversational AI assistant that uses Retrieval-Augmented Generation (RAG) to answer questions about Fiverr's Terms of Service documents.

## Overview

This project implements a chatbot that uses the LangChain framework to:
1. Process and store document content in a vector database
2. Retrieve relevant information based on user queries
3. Generate contextually relevant responses using a Large Language Model
4. Maintain conversation history for contextual awareness

## Features

- **Document Processing**: Automatically processes PDF documents and stores them in a vector database
- **Semantic Search**: Uses embeddings to find the most relevant document sections for user queries
- **Conversational Memory**: Maintains chat history for context-aware responses
- **Multiple Interfaces**: 
    - Command-line interface for direct interaction
    - Streamlit web application for a user-friendly experience

## Directory Structure

```
1_chat_models_starter.py             # Starter file for basic chat model implementation
2_conversational_chat_model.py       # Implementation of a conversational chat model
3_creating_vector_db.py              # Script to create the vector database from PDFs
4_retrieving_data_from_vector_db.py  # Script to retrieve data from the vector database
5_RAG_using_langchain.py             # Main RAG implementation using LangChain
app.py                               # Streamlit web application
chat_history.json                    # Saved chat history
processed_pdfs.json                  # Record of processed PDF files
readme.md                            # This file
chroma_db/                           # Vector database directory
pdf_docs/                            # Source PDF documents
```

## Requirements

- Python 3.8+
- LangChain
- Hugging Face Hub account with API key
- ChromaDB
- Streamlit (for web interface)
- Dotenv (for environment variables)

## Installation

1. Clone this repository:
     ```
     git clone <repository-url>
     cd LangChain/1_Chat_Models
     ```

2. Install dependencies:
     ```
     pip install langchain langchain_huggingface langchain_core chromadb streamlit python-dotenv
     ```

3. Create a `.env` file in the project root with your Hugging Face API key:
     ```
     HUGGING_FACE_API_KEY=your_hugging_face_api_key
     ```

## Usage

### Command Line Interface

Run the RAG chatbot from the command line:

```
python 5_RAG_using_langchain.py
```

Type your questions about Fiverr's Terms of Service and the chatbot will respond with relevant information. Type 'exit' to end the conversation.

### Web Interface

Launch the Streamlit web application:

```
streamlit run app.py
```

The web interface will allow you to:
- Ask questions about Fiverr's Terms of Service
- View the retrieved document sections used to answer your question
- See the entire conversation history

## Implementation Details

### LLM Model

The chatbot uses the Mistral-7B-Instruct-v0.1 model from Hugging Face:
- Temperature: 0.5 (balanced between creativity and accuracy)
- Max new tokens: 100 for CLI, 100 for web interface

### Retrieval Process

1. User query is converted to embeddings
2. Similar documents are retrieved from ChromaDB
3. Retrieved documents provide context for the LLM
4. LLM generates a response based on the retrieved context and conversation history

### Prompt Template

```
<s>[INST] You are a helpful AI assistant. Here are some documents that might help you answer the user's question. Keep responses according the context:
Context:
{context}

Current conversation:
{history}
Human: {input}
Assistant: [/INST]
```

## Files and Functions

- **5_RAG_using_langchain.py**: Core implementation of the RAG system
    - `initialize_llm()`: Sets up the Hugging Face model
    - `initialize_chroma_db()`: Connects to the ChromaDB vector database
    - `retrieve_relevant_documents()`: Searches for relevant document sections
    - `format_history_for_prompt()`: Formats conversation history

- **app.py**: Streamlit web interface
    - Provides a user-friendly chat interface
    - Displays retrieved documents alongside responses


