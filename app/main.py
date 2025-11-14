# import chainlit as cl
# from chainlit.server import add_route
from fastapi import FastAPI
from chainlit.utils import mount_chainlit


app = FastAPI()

mount_chainlit(
    app=app,
    path="/chat",           # URL you will open
    target='chainlit_script.py'    # Absolute path to chainlit script
)

# add_route(
#     app=app,
#     path="/chat",         # URL where Chainlit will be served
#     target='chainlit_script.py'  # Absolute path to chat.py
# )

# @cl.on_message
# async def main(message: cl.Message):
#     user_text = message.content
#     response = await llm.acomplete(user_text)
#     await cl.Message(response).send()

# -------------------------------
# 3. Mount Chainlit to FastAPI
# -------------------------------
# add_route(app, path="/")  # Chainlit will be available at http://localhost:8000/chat

# -------------------------------
# 4. A test endpoint (optional)
# -------------------------------
@app.get("/health")
def health():
    return {"status": "Running", "chainlit": "/chat"}