from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# read the current env variable
model_to_use = os.environ['MODEL_TO_USE']
if model_to_use == 'openai':
    # model definition
    api_key = os.environ['OPENAI_API_KEY']
    raw_llm_model = ChatOpenAI(
        model="gpt-5.1",
        api_key=api_key,
        temperature=0.5
    )
    # # Add initial prompt here
    # system_msg = SystemMessagePromptTemplate.from_template(
    #     "You are a helpful, polite, and professional assistant. "
    #     "Always answer clearly, respectfully, and concisely. "
    #     "If unsure, politely say you don’t have enough information. "
    #     # "Always answer in newyork slang by adding the word yo in the beginning of every sentence."
    # )
    # human_msg = HumanMessagePromptTemplate.from_template("{question}")
    # prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
    # llm_model = prompt | raw_llm_model
    # # embedding model here
    # embeddings_model = OpenAIEmbeddings(
    #     model="text-embedding-3-large",
    #     api_key=api_key
    # )
    # # pinecone client here
    # PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
    # pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    # pinecone_index = pinecone_client.Index('cipaca-docs')