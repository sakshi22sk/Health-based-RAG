from fastapi import FastAPI
from pydantic import BaseModel
from rag import generate_answer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
app = FastAPI(title="Healthcare RAG System")
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_healthcare_bot(request: QueryRequest):
    response = generate_answer(request.question)
    return {
        "question": request.question,
        "answer": response,
        "disclaimer": "This system is for educational purposes only."
    }
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("ui/index.html", "r", encoding="utf-8") as f:
        return f.read()