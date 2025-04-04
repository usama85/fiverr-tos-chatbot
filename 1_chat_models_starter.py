# from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the Mistral model from Hugging Face
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),  # Or load from .env
    temperature=0.5,
    max_new_tokens=50
    # repo_id="mistralai/Mistral-7B-Instruct-v0.1", 
    # model_kwargs={"temperature": 0.7, "max_new_tokens": 50},
    # huggingfacehub_api_token=os.getenv("HUGGING_FACE_API_KEY"),
)

response = llm.invoke("What is Retrieval-Augmented Generation (RAG)?")


# # Create a prompt template
# prompt = PromptTemplate(
#     input_variables=["what is the capital of France?"],
#     template="Answer the following question: {question}"
# )

# # Create an LLM chain
# llm_chain = LLMChain(llm=llm, prompt=prompt)

# # Run the chain with a sample question
# question = "What is Retrieval-Augmented Generation (RAG)?"
# response = llm_chain.run(question)

print(response)