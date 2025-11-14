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
from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentAction, AgentFinish
import uuid
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
    print('patient_id', patient_id)
    url = f"https://sia-api-test.cipaca.com/v1/patients/JKG1/62"
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOjEsImlhdCI6MTc2MjY4MzQ4NywiZXhwIjoxNzYyNjk0Mjg3LCJ0eXBlIjoiQUNDRVNTIn0.ZsqY7DJKyveJGblsRgdKsLns88Fi78HM0DiJe26zrB0"
    }
    response = requests.get(url, headers=headers)
    print('called the aptient personal info method')
    print(response)
    if response.status_code == 200:
        data = response.json()
        print(data)
        return f"Patient information Found: {data}"
    else:
        return "No data available or API error."

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# memory = ConversationBufferMemory(
#     llm=llm_model,             # Pass the LLM to the memory to summarize/manage history
#     memory_key="chat_history", # This must match the expected input key for the chain
#     return_messages=True ,      # Use a list of message objects for cleaner history passing,
#     output_key="answer"
# )

# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm=llm_model,
#     retriever=retriever,
#     # memory=memory,                     # Pass the initialized memory object
#     chain_type="stuff",
#     return_source_documents=True,
#     combine_docs_chain_kwargs={
#         "prompt": CUSTOM_QA_PROMPT
#     }
# )

qa_chain = RetrievalQA.from_chain_type(
    llm=raw_llm_model,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        # The prompt is passed here for the final answer synthesis
        # "prompt": CUSTOM_QA_PROMPT 
    }
)

def rag_func(input_data: Any):
    print("🔍 RAW TOOL INPUT:", input_data)
    # Normalize input (ensure plain string)
    if isinstance(input_data, dict):
        if "query" in input_data:
            input_data = input_data["query"]
        elif "input" in input_data:
            input_data = input_data["input"]
    return qa_chain.invoke({"query": str(input_data)})["result"]

# 1️⃣ Wrap your RAG chain as a Tool
rag_tool = Tool(
    name="medical_rag_reference",
    # func=lambda q: qa_chain.invoke({"question": q})["answer"],
    func=rag_func,
    description="Answers medical questions using CIPACA's internal reference documents."
)

# Define the Agent's Prompt
# AGENT_PROMPT = ChatPromptTemplate.from_messages([
#     ("system", "You are the CIPACA Clinical Triage Agent. Your primary goal is to use the available tools to answer complex medical and patient-specific queries. **Always prioritize checking for patient data using the `patient_personal_information` tool first if a patient ID is mentioned.** If the query is about medical concepts or protocols, use the `medical_rag_reference` tool."),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}"), # Crucial for the agent's thought process
# ])

# SYSTEM_MESSAGE_CONTENT = (
#     "You are the CIPACA Clinical Triage Agent. "
#     "Your primary goal is to use the available tools to answer complex medical and patient-specific queries. "
#     "**Always prioritize checking for patient data using the `patient_personal_information` tool first if a patient ID is mentioned."
#     "** If the query is about medical concepts or protocols, use the `medical_rag_reference` tool. If the query is about medical concepts or protocols, use the medical_rag_reference tool. When calling medical_rag_reference, the input must only be the concise question or topic (e.g., 'Explain shock' or 'AMI management protocol'). Do not include retrieved document content, internal thoughts, or prior answers as tool input."
# )

# SYSTEM_MESSAGE_CONTENT = (
#     "You are the CIPACA Clinical Agent."
#     "Your Primary goal is to use the available tools to answer complex medical and patient specific queries"
#     "**If the query is about general medical concepts or protocols, use the `medical_rag_reference_tool`. When calling `medical_rag_reference`, the input must only be the concise question or topic. Do not include retrieved document content, internal thoughts, or prior answers as tool input.**"
#     "**If the query is about a patient with their ID, then call `patient_personal_information` tool.**"
# )

SYSTEM_MESSAGE_CONTENT = (
    "You are the CIPACA Clinical Agent.\n"
    "Your job is to either:\n"
    "**Use the `patient_personal_information` tool information about a patient is asked. Patient ID is needed for the tool as an argument. Patient ID is an integer. Read the patient ID and pass it as an argument to the tool - `patient_personal_information`. Read the name of the patient from the output and respond back.**"
    "Always respond to the user directly with the final, concise answer after tool use.\n"
    "Do NOT greet, chat casually, or respond with generic phrases like 'Hello'."
)


tools = [get_patient_personal_information]

agent_runnable = create_agent(
    tools=tools,
    model=raw_llm_model,
    system_prompt=SYSTEM_MESSAGE_CONTENT # CRITICAL: Pass the Agent Prompt
    # verbose=True,
)


class AgentState(TypedDict):
    """Represents the state of the agent in the graph."""
    # List of all messages (conversation history + tool call history)
    messages: List[BaseMessage]


tool_map = {tool.name: tool.func for tool in tools}

def execute_tools(state):
    """
    Executes the tool call(s) generated by the LLM.
    This replaces the deprecated ToolExecutor logic.
    """
    latest_message = state["messages"][-1]
    
    # LangGraph ensures the latest message contains 'tool_calls' if we enter this node
    tool_calls = latest_message.tool_calls
    
    tool_results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # 1. Look up the function and execute
        tool_func = tool_map.get(tool_name)
        
        if tool_func:
            print(f"**--- TOOL CALL: {tool_name} with input: {tool_args} ---**")
            # Execute the function using Python's keyword arguments unpacking
            observation = tool_func(**tool_args)
            
            # 2. Append the result as a new message (Observation)
            tool_results.append(AIMessage(
                content=str(observation), # Tool result is the content
                # CRUCIAL: Link the observation back to the original tool call ID
                # Note: In LangChain/LangGraph, the tool call is often a dict
                tool_call_id=tool_call.get("id", "NoID"), 
            ))
        else:
            tool_results.append(AIMessage(
                content=f"Tool Error: Tool '{tool_name}' not found.",
                tool_call_id=tool_call.get("id", "NoID"),
            ))

    # Return the state with the new observation messages added
    return {"messages": state["messages"] + tool_results}


def call_model(state):
    """
    Runs the LLM agent, which can output a final answer or a tool call.
    Crucially, ensures output is always a BaseMessage type.
    """
    
    current_input = state["messages"][-1].content
    
    # Run the agent logic, which returns AgentFinish (a message) or AgentAction (a dict/object)
    response = agent_runnable.invoke(
        {"input": current_input, "agent_scratchpad": state["messages"][:-1]}
    )
    
    # ⚠️ FIX: Check if the response is an AgentAction (which often comes back as a dict/object 
    # from create_agent if it's not handled by a parser that converts it to BaseMessage).
    
    if isinstance(response, list) and response and isinstance(response[0], AgentAction):
        # Handle the case where the agent returns a list of actions (e.g., in legacy agents)
        action = response[0]
        # Convert the AgentAction to a list of dicts for tool_calls structure
        tool_calls = [
            {
                "name": action.tool,
                "args": action.tool_input,
                "id": str(uuid.uuid4()) # Generate a temporary ID for tracking
            }
        ]
        # Wrap the action in an AIMessage with tool_calls
        final_message = AIMessage(content="", tool_calls=tool_calls)
        
    elif isinstance(response, AgentAction):
        # Handle the common case: single AgentAction object
        tool_calls = [
            {
                "name": response.tool,
                "args": response.tool_input,
                "id": str(uuid.uuid4()) # Generate a temporary ID for tracking
            }
        ]
        final_message = AIMessage(content="", tool_calls=tool_calls)

    elif isinstance(response, BaseMessage):
        # If it's a message (like AgentFinish), use it directly
        final_message = response
        
    elif isinstance(response, dict) and 'tool' in response and 'tool_input' in response:
         # Handle the case where a raw dict is returned (like in your previous manual loop)
        tool_calls = [
            {
                "name": response['tool'],
                "args": response['tool_input'],
                "id": str(uuid.uuid4())
            }
        ]
        final_message = AIMessage(content="", tool_calls=tool_calls)
        
    else:
        # Fallback for unexpected output (treat as final text answer)
        final_message = AIMessage(content=str(response))
        
    return {"messages": state["messages"] + [final_message]}

def should_continue(state):
    """
    Conditional edge: Checks if the latest message is a tool call or the final answer.
    """
    latest_message = state["messages"][-1]
    
    # The agent output is wrapped as a BaseMessage; we check for tool_calls attribute
    if latest_message.tool_calls:
        return "continue" # Go to the 'tool' node
    else:
        return "end"     # End the graph

# Initialize the graph with the defined state
workflow = StateGraph(AgentState)

# Add the two core nodes
workflow.add_node("model", call_model)
workflow.add_node("tool", execute_tools)

# Set the entry point: Start by calling the model
workflow.set_entry_point("model")

# Define the primary conditional edge
workflow.add_conditional_edges(
    "model", 
    should_continue,
    {
        "continue": "tool", # If tool call is needed, go to the tool node
        "end": END          # If final answer is ready, end
    }
)

# Define the return edge: After a tool is executed, go back to the model 
# to incorporate the observation and generate the next step.
workflow.add_edge("tool", "model")

# Compile the final graph
cipaca_triage_agent = workflow.compile()


if __name__ == "__main__":

# 1. Test the Patient Data tool path
    patient_query = "What is the patient information for 62?"
    
    # # The graph is invoked with the initial message wrapped in the state dictionary
    # print(f"\n--- Running Agent for Patient Query: {patient_query} ---")
    
    # # The graph returns the final state
    # final_state_patient = cipaca_triage_agent.invoke({
    #     "messages": [HumanMessage(content=patient_query)]
    # })
    
    # # The final answer is the content of the last message in the list
    # final_answer_patient = final_state_patient["messages"][-1].content
    # print(f"\n**FINAL AGENT ANSWER (Patient Data):**\n{final_answer_patient}")
    
    # print("\n" + "="*50 + "\n")
    
    # # 2. Test the RAG tool path
    # rag_query = "Explain shock, as per our internal manuals."
    
    # print(f"--- Running Agent for RAG Query: {rag_query} ---")
    # final_state_rag = cipaca_triage_agent.invoke({
    #     "messages": [HumanMessage(content=rag_query)]
    # })
    
    # final_answer_rag = final_state_rag["messages"][-1].content
    # print(f"\n**FINAL AGENT ANSWER (Medical RAG):**\n{final_answer_rag}")