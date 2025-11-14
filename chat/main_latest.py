from .llm import raw_llm_model, pinecone_index, embeddings_model
from langchain_community.agents import create_tool_calling_agent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import requests
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA


# -------------------------------
# 1️⃣ Define Your QA Chain Tool
# -------------------------------

vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings_model,
    text_key="text"
)

retriever = vectorstore.as_retriever(search_kwargs={
       "k": 3
       })

qa_chain = RetrievalQA.from_chain_type(
    llm=raw_llm_model,
    retriever=retriever,
    chain_type="stuff",  # "map_reduce" or "refine" for long docs
    return_source_documents=True
)



class QAQuery(BaseModel):
    question: str = Field(description="The user's medical or technical question to search for.")

def qa_func(args) -> str:
    try:
        result = qa_chain.invoke({"query": args.question})
        # If you're using return_source_documents=True, extract result["result"]
        answer = result.get("result") if isinstance(result, dict) else result
        return answer or "No relevant information found in the knowledge base."
    except Exception as e:
        return f"Error while querying QA chain: {e}"


qa_tool = StructuredTool.from_function(
    func=qa_func,
    name="qa_chain_tool",
    description="Answers user questions using internal knowledge and embeddings."
)


# -------------------------------
# 2️⃣ Define Your API Call Tool
# -------------------------------

class PatientLookup(BaseModel):
    patient_id: str = Field(description="Unique patient identifier to fetch info for.")

def fetch_patient_data(args: PatientLookup) -> str:
    """
    Example API call. Replace URL and headers with your actual endpoint.
    """
    url = f"https://api.example.com/v1/patients/{args.patient_id}"
    headers = {"Authorization": "Bearer YOUR_TOKEN"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return f"Patient data: {response.json()}"
        else:
            return f"API returned {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error calling API: {e}"

api_tool = StructuredTool.from_function(
    func=fetch_patient_data,
    name="patient_api_tool",
    description="Fetches patient details from the hospital API using patient ID."
)


# -------------------------------
# 3️⃣ Create the Agent
# -------------------------------

system_prompt = """
You are a clinical assistant agent.
- Use the qa_chain_tool for knowledge questions.
- Use the patient_api_tool when the user mentions a patient ID.
Always give short, clear answers.
"""

agent = create_tool_calling_agent(
    llm=raw_llm_model,
    tools=[qa_tool, api_tool],
    system_prompt=system_prompt
)


# -------------------------------
# 4️⃣ Run the Agent
# -------------------------------

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    # Example 1 — Internal knowledge query
    print("\n--- Example: Knowledge Query ---")
    result = agent.invoke({"messages": [HumanMessage(content="Explain sepsis management")]})
    print(result.output_text)

    # Example 2 — API call
    print("\n--- Example: API Call ---")
    result = agent.invoke({"messages": [HumanMessage(content="Get details for patient ID 62")]})
    print(result.output_text)
