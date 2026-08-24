from pathlib import Path
import sys
from pinecone import Pinecone

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME


def upload_chunks_to_pinecone(chunks, batch_size=100):
    """
    Uploads a list of chunk dicts to Pinecone.
    Each chunk must have: filename, chunk_index, text, embedding.
    """
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    vectors = []
    for chunk in chunks:
        vector_id = f"{chunk['filename']}-chunk-{chunk['chunk_index']}"
        vectors.append({
            "id": vector_id,
            "values": chunk["embedding"],
            "metadata": {
                "text": chunk["text"],
                "filename": chunk["filename"]
            }
        })

    # Upsert in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Uploaded batch {i // batch_size + 1} ({len(batch)} vectors)")

    print(f"Successfully uploaded {len(vectors)} chunks to Pinecone.")
    return len(vectors)