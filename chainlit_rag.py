import chainlit as cl

# from chat._main import load_index, make_qa_chain
from chat.main_rag import qa_chain


@cl.on_chat_start
async def start_chat():
    """Load the QA chain once per session and greet the user."""
    current_role = "Intensivist"
    cl.user_session.set("qa_chain", qa_chain)
    await cl.Message(
        content=(
            "I’m your AI medical reference {}. You can ask me about clinical "
            "guidelines, drug interactions, diagnostic protocols, or patient "
            "management summaries.".format(current_role)
        )
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")
    if qa_chain is None:
        await cl.Message(
            content="I’m still loading the knowledge base. Please try again in a moment."
        ).send()
        return

    # 2. **CRITICAL FIX**: Change input key from "query" to "question"
    # The ConversationalRetrievalChain expects "question" as the input key.
    result = await cl.make_async(qa_chain.invoke)({"question": message.content})
    
    # The output key for the answer is 'answer' for ConversationalRetrievalChain
    answer = result.get("answer", "") 
    source_documents = result.get("source_documents", [])

    if not answer:
        answer = "I couldn’t find anything relevant. Could you rephrase your question?"

    # 3. Handle Source Documents (Retrieval Mechanism)
    if source_documents:
        # Create a list of Chainlit Elements for the sources
        elements = [
            cl.Text(
                name=f"Source {i+1}", 
                content=doc.page_content, 
                display="side",
                # Include metadata if available (e.g., 'source' file path)
                # metadata={"source": doc.metadata.get("source", "N/A")} 
            )
            for i, doc in enumerate(source_documents)
        ]
        
        # Send the final answer with source elements attached
        await cl.Message(content=answer, elements=elements).send()
    else:
        # Send the answer without sources
        await cl.Message(content=answer).send()
