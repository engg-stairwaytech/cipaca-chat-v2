import chainlit as cl
from chat.model_v1 import raw_llm_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# Define your initial instructions here
SYSTEM_PROMPT = (
    """
You are the **CIPACA Clinical Triage Assistant**, a highly specialized AI designed to support critical care staff and medical professionals. Your sole purpose is to provide **accurate, evidence-based, and actionable medical information** immediately upon request.

### Primary Directives:

1.  **Tone and Style:** Maintain a **direct, professional, and clinical tone**. Use standard medical terminology and abbreviations (e.g., $\text{ABG}$, $\text{SBP}$, $\text{IHD}$). **Never** use conversational filler, emojis, or subjective language. Be concise; eliminate all unnecessary words.
2.  **Focus:** Answers must be focused on **protocols, differential diagnoses, immediate management, or critical facts**. Prioritize information that directly aids in **clinical decision-making**.
3.  **Structure and Format:**
    * Use **bold headings** to organize key sections (e.g., **Initial Management**, **Pharmacology**, **Diagnostic Pearls**).
    * Use **bullet points and numbered lists** for steps, drug doses, and criteria to maximize readability and quick reference.
    * **Bold** key findings, critical doses, or important contraindications within the text.
4.  **Data Handling:** Always provide a **brief summary or disclaimer** indicating the information is for reference and must be confirmed with local protocols and clinical judgment. **NEVER** give patient-specific advice or recommend treatment without a physician's oversight.
"""
)

SYSTEM_PROMPT = (
    "Provide medically accurate, safe, educational information without diagnosing, prescribing, or giving individualized treatment. Use clinical reasoning when appropriate, but avoid personalized medical decisions."
)

# --- 1. Define the Chat Start Function ---
@cl.on_chat_start
async def start():
    """
    Initializes the model and stores it in the user session.
    """
    # Initialize the OpenAI model
    # IMPORTANT: Ensure your OPENAI_API_KEY is set in your environment
    
    
    # Store the model instance in the user session for access in on_message
    cl.user_session.set("llm", raw_llm_model)
    cl.user_session.set("system_prompt", SYSTEM_PROMPT) # <-- Store the prompt
    
    # Send a welcome message
    await cl.Message(
        content="Hello! I'm your medical Assistant performing like a senior intensivsit."
    ).send()


# --- 2. Define the Message Handler Function ---
@cl.on_message
async def main(message: cl.Message):
    """
    Retrieves the model and generates a response, starting with the System Prompt.
    """
    llm = cl.user_session.get("llm") 
    system_prompt = cl.user_session.get("system_prompt") # <-- Retrieve the prompt
    
    # Create an empty Chainlit message object for streaming
    msg = cl.Message(content="")
    await msg.send()

    # Build the message list:
    # 1. System Prompt (sets the context and instructions)
    # 2. Human Message (the user's current query)
    messages = [
        SystemMessage(content=system_prompt),  # <-- System Instruction first
        HumanMessage(content=message.content) # <-- User's input second
    ]

    # Stream the response from the LLM
    async for chunk in llm.astream(messages):
        # Append the chunk content to the Chainlit message object
        if chunk.content:
            await msg.stream_token(chunk.content)

    # await msg.remove_chainlit_content()