from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from chainlit.utils import mount_chainlit

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


mount_chainlit(app=app, target="chainlit_rag.py", path="/chainlit")