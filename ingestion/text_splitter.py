def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split a long string into chunks of `chunk_size` characters,
    with `overlap` characters between consecutive chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks


def chunk_pdf_texts(pdf_texts, chunk_size=500, overlap=50):
    """
    Accepts a list of (filename, text) tuples.
    Returns a list of dicts with keys: filename, chunk_index, text.
    """
    all_chunks = []
    for filename, text in pdf_texts:
        chunks = chunk_text(text, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": filename,
                "chunk_index": idx,
                "text": chunk
            })
    return all_chunks