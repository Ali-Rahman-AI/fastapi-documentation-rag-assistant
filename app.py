"""
app.py

FastAPI application entry point. Exposes:

    GET  /        -> serves the web interface
    GET  /health  -> simple health check
    POST /ask     -> accepts a question and returns a grounded answer

Run with:
    python app.py
or
    uvicorn app:app --reload
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
from rag import RAGPipeline
from utils import get_logger, clean_question

logger = get_logger(__name__)

app = FastAPI(
    title="FastAPI Documentation Assistant",
    description="A Retrieval-Augmented Generation assistant that answers "
                 "questions using only the official FastAPI documentation.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# The RAG pipeline is built once at startup and reused for every request.
rag_pipeline: RAGPipeline | None = None


class QuestionRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    """Build the RAG pipeline (embeddings + vector store + LLM client)."""
    global rag_pipeline
    logger.info("Application starting up...")
    try:
        rag_pipeline = RAGPipeline()
        if rag_pipeline.is_ready():
            logger.info("RAG pipeline is ready.")
        else:
            logger.warning(
                "RAG pipeline started but is NOT fully ready. "
                "Check GROQ_API_KEY and run 'python ingest.py' if needed."
            )
    except Exception as error:
        logger.error("Failed to initialize RAG pipeline: %s", error)
        rag_pipeline = None


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health_check():
    """Report whether the assistant is fully ready to answer questions."""
    ready = rag_pipeline is not None and rag_pipeline.is_ready()
    return {
        "status": "ok" if ready else "not_ready",
        "vector_db_ready": rag_pipeline.vector_store is not None if rag_pipeline else False,
        "llm_ready": rag_pipeline.llm_client is not None if rag_pipeline else False,
    }


@app.post("/ask")
def ask_question(payload: QuestionRequest):
    """Answer a question using only the indexed FastAPI documentation."""
    question = clean_question(payload.question)
    logger.info("Received question: %s", question)

    if not question:
        return JSONResponse(
            status_code=400,
            content={"answer": "Please type a question before submitting.", "sources": []},
        )

    if rag_pipeline is None or not rag_pipeline.is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "answer": "The assistant is not ready yet. Make sure the "
                          "documentation has been indexed and the API key is set.",
                "sources": [],
            },
        )

    try:
        result = rag_pipeline.answer_question(question)
        return JSONResponse(status_code=200, content=result)
    except ValueError as error:
        return JSONResponse(status_code=400, content={"answer": str(error), "sources": []})
    except RuntimeError as error:
        logger.error("Pipeline error: %s", error)
        return JSONResponse(status_code=502, content={"answer": str(error), "sources": []})
    except Exception as error:
        logger.error("Unexpected error while answering question: %s", error)
        return JSONResponse(
            status_code=500,
            content={
                "answer": "Something went wrong while processing your question. Please try again.",
                "sources": [],
            },
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.APP_HOST, port=config.APP_PORT, reload=True)
