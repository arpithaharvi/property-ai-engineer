# ============================================================
# modules/pdf_loader.py — PDF Loading and Groq Vision
# Handles: digital PDFs, scanned PDFs, Kannada + English
# Used by: app.py (Upload Documents page)
# ============================================================

import base64
import time
import io
import fitz
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    GROQ_VISION_MODEL,
    VISION_MAX_TOKENS,
    VISION_IMAGE_QUALITY,
    VISION_PAGE_ZOOM,
    DIRECT_TEXT_MIN_CHARS,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from modules.llm import load_groq_client

groq_client = load_groq_client()


def extract_text_with_groq_vision(page_image_pil, page_num):
    # Convert PIL image to base64 JPEG
    img_byte_arr = io.BytesIO()
    page_image_pil.save(
        img_byte_arr,
        format='JPEG',
        quality=VISION_IMAGE_QUALITY
    )
    image_base64 = base64.b64encode(
        img_byte_arr.getvalue()
    ).decode('utf-8')

    # Retry on rate limit: waits 10s → 20s → 30s
    max_retries = 3
    wait_times = [10, 20, 30]

    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This is a page from an Indian property "
                                    "document — sale deed, valuation report, "
                                    "or title document. "
                                    "Extract ALL visible text exactly as written. "
                                    "Include English and Kannada text. "
                                    "Preserve tables, field names, values, "
                                    "owner names, addresses, boundary details, "
                                    "survey numbers, and all other text. "
                                    "Do not summarize — extract everything."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=VISION_MAX_TOKENS
            )
            text = response.choices[0].message.content
            has_kannada = any(
                '\u0C80' <= c <= '\u0CFF' for c in text
            )
            return text, has_kannada

        except Exception as e:
            is_rate_limit = any(x in str(e).lower() for x in [
                "rate_limit", "429", "too many requests", "413"
            ])
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(wait_times[attempt])
            else:
                return f"[Page {page_num} failed: {str(e)}]", False

    return f"[Page {page_num}: all retries failed]", False


def process_pdf(pdf_path, filename, progress_bar, status_text):
    # Auto-detects digital vs scanned pages
    # Digital → direct text extraction
    # Scanned → Groq Vision AI

    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    all_documents = []
    digital_count = 0
    vision_count = 0

    for page_num in range(total_pages):
        progress_bar.progress((page_num + 1) / total_pages)
        status_text.text(f"Page {page_num + 1} of {total_pages}...")

        page = pdf_document[page_num]
        direct_text = page.get_text().strip()

        if len(direct_text) > DIRECT_TEXT_MIN_CHARS:
            digital_count += 1
            has_kannada = any(
                '\u0C80' <= c <= '\u0CFF' for c in direct_text
            )
            doc = Document(
                page_content=direct_text,
                metadata={
                    "source": filename,
                    "page": page_num + 1,
                    "language": "kannada+english" if has_kannada else "english",
                    "extraction_method": "direct_text"
                }
            )
            all_documents.append(doc)

        else:
            vision_count += 1
            matrix = fitz.Matrix(VISION_PAGE_ZOOM, VISION_PAGE_ZOOM)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples
            )
            extracted_text, has_kannada = extract_text_with_groq_vision(
                img, page_num + 1
            )
            if len(extracted_text.strip()) > 20:
                doc = Document(
                    page_content=extracted_text,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,
                        "language": "kannada+english" if has_kannada else "english",
                        "extraction_method": "groq_vision"
                    }
                )
                all_documents.append(doc)
            time.sleep(1)

    pdf_document.close()
    return all_documents, digital_count, vision_count


def split_into_chunks(documents):
    # Split documents into chunks for ChromaDB
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)
