import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from PIL import Image

from langchain.schema import Document

# Initialize models
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

# OCR fallback function
def extract_text_paddle(image):
    result = ocr_model.ocr(image, cls=True)
    lines = [line[1][0] for line in result[0]]
    return "\n".join(lines)

# Paragraph splitter and merger

def split_text_paragraphs(text, max_length=1000):
    # Split on double newlines
    raw_paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]

    # Merge small paragraphs to reduce chunk count
    merged_chunks = []
    current = ""
    for para in raw_paragraphs:
        if len(current) + len(para) < max_length:
            current = f"{current}\n{para}".strip()
        else:
            merged_chunks.append(current.strip())
            current = para
    if current:
        merged_chunks.append(current.strip())

    return merged_chunks

# Process a single PDF and return LangChain Documents
def process_pdf(pdf_path):
    docs = []
    filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc):
        text = page.get_text()

        if not text or len(text.strip()) < 50:
            try:
                images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1, dpi=300)
                if not images:
                    raise ValueError("No image generated from PDF page.")
                text = extract_text_paddle(images[0])
                if not text or len(text.strip()) < 10:
                    text = "[Unable to extract text from scanned image.]"
            except Exception as e:
                print(f"[❌ Error processing page {page_num+1} of {filename}]: {e}")
                text = "[Unable to extract text due to error.]"

        chunks = split_text_paragraphs(text)
        for i, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "page": page_num + 1,
                    "chunk_id": i
                }
            ))
    return docs
