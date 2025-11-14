import chainlit as cl

# from chat._main import load_index, make_qa_chain
from chat.main import qa_chain, agent_runnable, tools, cipaca_triage_agent
from langchain_core.messages import AIMessage, HumanMessage
import uuid

AGENT_CONFIG = {
    "agent_runnable": agent_runnable,
    "tools": tools,
}


@cl.on_chat_start
async def start():
    # Store the compiled agent graph in the user session for later use
    cl.user_session.set("agent_graph", cipaca_triage_agent)
    
    # Send a welcome message
    await cl.Message(
        content="🩺 **CIPACA Clinical Triage Agent Initialized!** 🚀\n"
                "I can fetch patient data (via API) or answer medical questions (via RAG).\n"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Send a simple indicator
    await cl.Message(content="🩺 Running CIPACA agent...").send()
    
    agent_graph = cl.user_session.get("agent_graph")
    initial_message = HumanMessage(content=message.content)
    
    # 1. Capture the final state from the synchronous invocation
    # We do NOT assign this result to any temporary message or variable 
    # that might be implicitly logged.
    final_state = await cl.make_async(agent_graph.invoke)(
        {"messages": [initial_message]}
    )
    
    # 2. Process and send
    final_answer_message = final_state["messages"][-1]
    final_answer_content = str(final_answer_message.content)
    
    # 3. Send the final answer as a NEW message
    await cl.Message(
        content=final_answer_content,
        # Setting a unique ID might isolate it from verbose logging
        id=str(uuid.uuid4()) 
    ).send()
    
    # 4. Return immediately to prevent any lingering final_state object from being logged
    return