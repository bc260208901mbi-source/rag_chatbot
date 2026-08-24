from sentence_transformers import SentenceTransformer

# Use the same model as in your chatbot
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks):
    """
    Accepts a list of chunk dicts (each with 'text' key).
    Returns the same list with an added 'embedding' key.
    """
    for chunk in chunks:
        chunk["embedding"] = model.encode(chunk["text"]).tolist()
    return chunks