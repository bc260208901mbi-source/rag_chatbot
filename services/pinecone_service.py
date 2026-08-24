from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def query_pinecone(embedding, top_k=3):
    """
    Query Pinecone with the embedding vector.
    Returns a list of relevant text chunks.
    """
    result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )

    matches = result.matches if hasattr(result, "matches") else result.get("matches", [])

    contexts = []

    for match in matches:
        metadata = match.metadata if hasattr(match, "metadata") else match.get("metadata", {})

        if isinstance(metadata, dict):
            text = metadata.get("text", "")
        else:
            text = ""

        if text:
            contexts.append(text)

    return contexts