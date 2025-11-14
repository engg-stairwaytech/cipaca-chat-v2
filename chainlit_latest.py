import chainlit as cl
from langchain_core.messages import HumanMessage
from chat.main_latest import agent

@cl.on_chat_start
async def start():
    cl.user_session.set("agent", agent)
    await cl.Message(content="👋 Hello! I'm the CIPACA Assistant. How can I help you today?").send()


@cl.on_message
async def handle_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    if not agent:
        await cl.Message(content="⚠️ Agent not initialized. Please restart the session.").send()
        return

    try:
        result = await cl.make_async(agent.invoke)({"messages": [HumanMessage(content=message.content)]})
        await cl.Message(content=result.output_text).send()
    except Exception as e:
        await cl.Message(content=f"❌ Error: {e}").send()