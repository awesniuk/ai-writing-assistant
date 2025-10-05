from fastapi import FastAPI
from pydantic import BaseModel

from backend.model import llama_model
from backend.rag import retrieve_similar_context
from backend.schemas import SaveRequest
from backend.db import save_document_to_db, init_db

app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_db()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate/")
def generate_text(data: PromptRequest):
    context = retrieve_similar_context(data.prompt)
    combined_prompt = "\n".join(context) + "\n\n" + data.prompt
    output = llama_model.generate(combined_prompt)
    return {"output": output}

@app.post("/save/")
def save_document(doc: SaveRequest):
    doc_id = save_document_to_db(doc.title, doc.content, doc.version)
    return {"status": "success", "doc_id": doc_id}