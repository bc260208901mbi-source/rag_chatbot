import os
import sys
from pdf_loader import load_pdfs_from_directory
from text_splitter import chunk_pdf_texts
from embedder import embed_chunks
from pinecone_uploader import upload_chunks_to_pinecone

# Ensure we can import config.py from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PINECONE_API_KEY, PINECONE_INDEX_NAME  # noqa: E402


def save_chunks_to_txt(chunks, output_file="chunks_output.txt"):
    """Write all chunks to a text file."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Total chunks: {len(chunks)}\n\n")
        for chunk in chunks:
            f.write(f"--- {chunk['filename']} | Chunk {chunk['chunk_index']} ---\n")
            f.write(chunk["text"] + "\n\n")
    print(f"Saved {len(chunks)} chunks to {output_file}")


def main():
    # 1. Directory containing your PDFs
    pdf_directory = input("Enter the path to the directory containing PDFs: ").strip()
    if not os.path.isdir(pdf_directory):
        print("Directory not found.")
        return

    # 2. Load PDFs
    print("\nLoading PDFs...")
    pdf_texts = load_pdfs_from_directory(pdf_directory)
    if not pdf_texts:
        print("No PDF files found.")
        return

    print(f"Loaded {len(pdf_texts)} PDF(s).")

    # 3. Create chunks
    chunk_size = 500
    overlap = 50
    print(f"\nChunking text (size={chunk_size}, overlap={overlap})...")
    chunks = chunk_pdf_texts(pdf_texts, chunk_size, overlap)
    print(f"Created {len(chunks)} chunks.")

    # 4. Save chunks to a txt file
    save_chunks_to_txt(chunks, "chunks_output.txt")

    # 5. Generate embeddings
    print("\nGenerating embeddings...")
    chunks = embed_chunks(chunks)

    # 6. Upload to Pinecone
    print("\nUploading to Pinecone...")
    uploaded_count = upload_chunks_to_pinecone(chunks)
    print(f"\nDone. Uploaded {uploaded_count} vectors to Pinecone index '{PINECONE_INDEX_NAME}'.")


if __name__ == "__main__":
    main()