import os
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    """Extract all text from a single PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def load_pdfs_from_directory(directory_path):
    """
    Load all PDF files from a directory.
    Returns a list of (filename, extracted_text) tuples.
    """
    pdf_texts = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(directory_path, filename)
            print(f"Processing: {filename}")
            text = extract_text_from_pdf(full_path)
            pdf_texts.append((filename, text))
    return pdf_texts