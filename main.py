from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.schemas import ChatRequest, ChatResponse
from services.embedding_service import get_embedding
from services.pinecone_service import query_pinecone
from services.llm_service import generate_answer

app = FastAPI(title="Simple RAG Chatbot")

@app.get("/")
def root():
    return {"status": "ok", "message": "RAG Chatbot API is running"}


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 1. Convert question to embedding
    embedding = get_embedding(request.question)

    # 2. Retrieve relevant context from Pinecone
    contexts = query_pinecone(embedding)

    # 3. Generate final answer with Groq
    answer = generate_answer(request.question, contexts)

    return ChatResponse(answer=answer)