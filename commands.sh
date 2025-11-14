chainlit run chainlit_ui.py -w

# Rag flow
chainlit run chainlit_rag.py -w


uvicorn app.main:app --reload --port 8000