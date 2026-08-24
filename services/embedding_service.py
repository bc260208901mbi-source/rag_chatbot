from sentence_transformers import SentenceTransformer

# Hugging Face sentence transformer model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def get_embedding(text: str):
    """
    Convert text into an embedding vector.
    """
    return model.encode(text).tolist()