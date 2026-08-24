import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Pinecone hosted embedding model
MODEL_NAME = "llama-text-embed-v2"


def get_embedding(text: str) -> list:
    """
    Generate embedding using Pinecone's hosted model.
    """
    try:
        result = pc.inference.embed(
            model=MODEL_NAME,
            inputs=[text],
            parameters={
                "input_type": "passage"
            }
        )

        return result.data[0].values

    except Exception as e:
        print(f"Embedding generation failed: {e}")
        raise