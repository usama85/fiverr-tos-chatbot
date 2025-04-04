from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
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

# Initialize the Mistral model using HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",  # Model repository ID
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),  # API token from environment
    temperature=0.5,  # Controls randomness in responses
    max_new_tokens=50  # Maximum number of tokens to generate
)

# Save chat history to JSON file
def save_chat_history(history, filename="chat_history.json"):
    with open(filename, "w") as f:
        json.dump(history, f, indent=2)

# Load chat history from JSON file
def load_chat_history(filename="chat_history.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Convert history list to string format for the prompt
def format_history_for_prompt(history_list):
    formatted_history = ""
    for message in history_list:
        formatted_history += f"Human: {message['user']}\nAssistant: {message['bot']}\n"
    return formatted_history

# Define the conversation prompt template
template = """<s>[INST] You are a helpful AI assistant. Keep responses super concise.
Current conversation:
{history}
Human: {input}
Assistant: [/INST]"""

# Create a PromptTemplate object with the template and input variables
prompt = PromptTemplate(template=template, input_variables=["history", "input"])

# Create a conversational chain using the LLM and the prompt
conversation_chain = LLMChain(llm=llm, prompt=prompt)

# Load chat history from file or start with an empty list
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
    
    # Format history for the prompt
    formatted_history = format_history_for_prompt(chat_history)
    
    # Generate AI response using the conversation chain
    response = conversation_chain.run(history=formatted_history, input=user_input)
    
    # Update conversation history
    chat_history.append({"user": user_input, "bot": response})
    
    # Save chat history after each interaction
    save_chat_history(chat_history)
    
    # Print the AI's response
    print(f"\nAI: {response}\n")

# Print the entire conversation history at the end
for i, message in enumerate(chat_history):
    print(f"--- Message {i+1} ---")
    print(f"User: {message['user']}")
    print(f"Bot: {message['bot']}")
    print()