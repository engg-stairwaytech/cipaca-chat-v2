from .llm import embeddings_model, llm_model, pinecone_client, pinecone_index, raw_llm_model
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool, Tool
import requests
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda 
from langchain_core.messages import AIMessage, HumanMessage
from typing import List, Tuple, Dict, Any
from langchain_core.exceptions import OutputParserException
from langchain_core.agents import AgentFinish
# from langchain_core.agents import AgentExecutor

vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings_model,
    text_key="text"
)

CUSTOM_QA_PROMPT = PromptTemplate.from_template("""
You are a professional AI assistant specialized in medical and academic knowledge to serve the organisation CIPACA. Your goal is to provide highly accurate, concise, and evidence-based medical information that is immediately useful in a critical care organisation CIPACA for saving lives. 
**Use medical terminology, abbreviations (e.g., ABG, Hgb, SBP), and a direct, professional, and action-oriented clinical tone.**"
CIPACA (Chennai Interventional Pulmonology and Critical Care Associates) is a healthcare organization that sets up and manages affordable, 24×7 ICU and critical care units in rural and semi-urban hospitals across India.
                         
                                                
**Formatting Rules:** 
    1. **Bolding for Headers:** Always use bold section headings (e.g., '**Summary:**') to structure the response. 
    2. **Bolding for Content:** Highlight critical data, key findings, or important medical terms within paragraphs by making them bold.
    3. **Extra Information:** Differentiate supplemental or extra clinical detail using *italics* or single-backtick code blocks (e.g., `` `lab value: 1.2` ``).
    4. **Section Separation:** End each major section with a markdown horizontal rule (e.g., '---') to show completion." # Adds lines
    5. **Visual Explanation:** Add relevant and professional emojis (e.g., 🌡️, 💊, ⚠️, 🩺) to make the answer more explanatory and visually engaging.
    6. **Disclaimer:** Conclude the entire answer with the following phrase in *italics* on a new line: '*Disclaimer: Always confirm findings with local protocol and clinical judgement. This information is for reference only and does not substitute for professional medical judgment.*'

Always answer with reasoning and references when available. 
When providing an answering with a question always, provide a compliment at the beginning of the answer for the user how good they were in questioning about the query and it can help CIPACA to save lives.
Use numbers and bullet points to highlight wherever needed. Use tables wherever needed to simplify the data.
After answering the question, provide a highlevel summary of the anwser either in a table or in a few sentences.
At the end, ask user if they want a followup question or more detailed explanation about that topic.

                                                
Conversation history:
{chat_history}

User question:
{question}

Relevant context from documents:
{context}

Now, based on the context above, provide a concise, factual, and well-structured answer.
If you don't know the answer, say "The available sources do not provide that information."

Answer:
""")

@tool("patient_personal_information", return_direct=False)
def get_patient_personal_information(patient_id) -> str:
    """Fetch a specific patient's personal information."""
    print('this method was called')
    url = f"https://sia-api-test.cipaca.com/v1/patients/JKG1/{patient_id}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return f"Patient information Found: {data}"
    else:
        return "No data available or API error."

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


memory = ConversationBufferMemory(
    llm=llm_model,             # Pass the LLM to the memory to summarize/manage history
    memory_key="chat_history", # This must match the expected input key for the chain
    return_messages=True ,      # Use a list of message objects for cleaner history passing,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_model,
    retriever=retriever,
    memory=memory,                     # Pass the initialized memory object
    chain_type="stuff",
    return_source_documents=True,
    combine_docs_chain_kwargs={
        "prompt": CUSTOM_QA_PROMPT
    }
)

# 1️⃣ Wrap your RAG chain as a Tool
rag_tool = Tool(
    name="medical_rag_reference",
    func=lambda q: qa_chain.invoke({"question": q})["answer"],
    description="Answers medical questions using CIPACA's internal reference documents."
)

# Define the Agent's Prompt
AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the CIPACA Clinical Triage Agent. Your primary goal is to use the available tools to answer complex medical and patient-specific queries. **Always prioritize checking for patient data using the `patient_personal_information` tool first if a patient ID is mentioned.** If the query is about medical concepts or protocols, use the `medical_rag_reference` tool."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), # Crucial for the agent's thought process
])

SYSTEM_MESSAGE_CONTENT = (
    "You are the CIPACA Clinical Triage Agent. Your primary goal is to use the available tools to answer complex medical and patient-specific queries. "
    "**Always prioritize checking for patient data using the `patient_personal_information` tool first if a patient ID is mentioned.** If the query is about medical concepts or protocols, use the `medical_rag_reference` tool."
    "If you have already called the appropriate tool and received its result, respond directly to the user with a final answer instead of calling more tools."
)

tools = [rag_tool, get_patient_personal_information]

agent_runnable = create_agent(
    tools=tools,
    model=raw_llm_model,
    system_prompt=SYSTEM_MESSAGE_CONTENT # CRITICAL: Pass the Agent Prompt
    # verbose=True,
)

# agent_executor = AgentExecutor(
#     agent=agent_runnable,
#     tools=tools,
#     verbose=True,
#     handle_parsing_errors=True
# )
# 💡 Define the Agent Executor REPLACEMENT
def execute_agent_loop(agent_runnable, tools, input_data, max_iterations=10):
    """
    Simulates the core AgentExecutor loop manually since the class is missing.
    Takes input, runs the agent, calls a tool if needed, and repeats.
    """
    
    # The 'agent_scratchpad' is critical for the agent's thought process
    scratchpad = []
    
    # ⚠️ NOTE: This function requires your raw LLM (llm_model) to be passed,
    # but for simplicity, we assume the agent_runnable already binds the tools.
    
    for _ in range(max_iterations):
        # 1. Prepare the input data for the agent runnable
        full_input = {
            "input": input_data["input"],
            "agent_scratchpad": scratchpad,
            # Note: If your prompt requires 'tools', you might need to add it here.
        }
        
        # 2. Run the agent logic (which is a Runnable)
        try:
            agent_output = agent_runnable.invoke(full_input)
        except OutputParserException as e:
            # Handle cases where the LLM fails to structure the tool call correctly
            scratchpad.append(f"Observation: Tool call failed due to parsing error: {e}")
            continue

        # 3. Check the output type
        if isinstance(agent_output, AgentFinish):
            # Agent wants to return the final answer
            return {"output": agent_output.return_values['output']}
        
        # If not AgentFinish, it must be an AgentAction (or list of actions)
        
        # Simplify to handle a single action for demonstration
        action = agent_output.pop() if isinstance(agent_output, list) else agent_output

        # Check if the output is a dictionary and contains the necessary keys
        if isinstance(action, dict) and 'tool' in action and 'tool_input' in action:
            # 4. Find and execute the tool
            tool_name = action['tool']  # Access key in dictionary
            tool_input = action['tool_input'] # Access key in dictionary
        else:
            # Handle unexpected output format (e.g., if the LLM output a thought or raw text)
            scratchpad.append(f"Observation: Agent output was not a recognized tool action: {action}")
            continue # Go to the next iteration
        
        # 4. Find and execute the tool
        tool_name = action.tool
        tool_input = action.tool_input
        
        # Find the correct function in the tools list
        tool_func = next((t.func for t in tools if t.name == tool_name), None)
        
        if tool_func:
            # Execute the tool
            print(f"**--- TOOL CALL: {tool_name} with input: {tool_input} ---**")
            observation = tool_func(tool_input)
            
            # 5. Append Observation to scratchpad for the next loop
            scratchpad.append(f"Tool Result: {observation}")
        else:
            scratchpad.append(f"Tool Error: Tool '{tool_name}' not found.")
            
    # If max iterations reached without finish
    return {"output": "Error: Maximum thinking iterations reached without providing a final answer."}

if __name__ == "__main__":

    query = "Who wrote the book 'The Washington Manual of Critical Care'"
    result = qa_chain.invoke({"question": query})

    print("ANSWER:\n", result)