from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import tempfile
import uuid

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
# Each session_id maps to {"chain": ..., "sources": [...]}
sessions: dict = {}

# ── Shared resources (loaded once) ───────────────────────────────────────────
embeddings = None

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def build_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return chain


# ── Request / Response models ─────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    urls: list[str]
    session_id: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    question: str
    system_context: Optional[str] = ""  # carries capability preferences from frontend


class IngestResponse(BaseModel):
    session_id: str
    message: str
    sources: list[str]
    chunk_count: int


class ChatResponse(BaseModel):
    answer: str
    source_docs: list[str]
    session_id: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _upsert_session(session_id: str, new_docs: list, source_label: str):
    """Add docs to an existing session's vectorstore or create a new one."""

    global embeddings

    if embeddings is None:
        print("Loading embeddings...", flush=True)
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        print("Embeddings loaded!", flush=True)


    chunks = text_splitter.split_documents(new_docs)
    if session_id in sessions:
        sessions[session_id]["vectorstore"].add_documents(chunks)
        sessions[session_id]["sources"].append(source_label)
        # Rebuild chain with updated vectorstore
        sessions[session_id]["chain"] = build_chain(sessions[session_id]["vectorstore"])
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        sessions[session_id] = {
            "vectorstore": vectorstore,
            "chain": build_chain(vectorstore),
            "sources": [source_label],
        }
    return len(chunks)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "RAG Chatbot API is running"}

from langchain_community.document_loaders import PyPDFLoader
@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    session_id: Optional[str] = None,
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = session_id or str(uuid.uuid4())

    # Save upload to temp file (PyPDFLoader needs a path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    if not docs:
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    chunk_count = _upsert_session(session_id, docs, file.filename)

    return IngestResponse(
        session_id=session_id,
        message=f"Ingested '{file.filename}' successfully.",
        sources=sessions[session_id]["sources"],
        chunk_count=chunk_count,
    )

from langchain_community.document_loaders import WebBaseLoader
@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(body: IngestURLRequest):
    session_id = body.session_id or str(uuid.uuid4())
    total_chunks = 0
    ingested = []

    for url in body.urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            total_chunks += _upsert_session(session_id, docs, url)
            ingested.append(url)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to load {url}: {str(e)}")

    return IngestResponse(
        session_id=session_id,
        message=f"Ingested {len(ingested)} URL(s) successfully.",
        sources=sessions[session_id]["sources"],
        chunk_count=total_chunks,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    if body.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please ingest documents first.")

    chain = sessions[body.session_id]["chain"]

    # Prepend capability instructions from the frontend if present
    full_question = (
        f"{body.system_context}\n\nQuestion: {body.question}"
        if body.system_context
        else body.question
    )

    result = chain.invoke({"question": full_question})

    # Extract unique source filenames / URLs from returned docs
    source_docs = list({
        doc.metadata.get("source", "unknown")
        for doc in result.get("source_documents", [])
    })

    return ChatResponse(
        answer=result["answer"],
        source_docs=source_docs,
        session_id=body.session_id,
    )


@app.get("/session/{session_id}")
def get_session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id": session_id,
        "sources": sessions[session_id]["sources"],
    }


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session cleared."}