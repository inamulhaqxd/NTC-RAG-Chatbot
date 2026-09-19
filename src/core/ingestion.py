"""PDF ingestion — extract text from PDFs."""

import logging
import re
from pathlib import Path

import pymupdf4llm

log = logging.getLogger(__name__)


def clean_text(text):
    """Remove image tags and extra whitespace."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_pymupdf(pdf_path):
    """Extract text using PyMuPDF4LLM (fast, good for text PDFs)."""
    return pymupdf4llm.to_markdown(str(pdf_path))


def extract_text_ocr(pdf_path):
    """Extract text using Docling OCR (for scanned PDFs)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import OcrAutoOptions, OcrMode, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=OcrAutoOptions(mode=OcrMode.PDF_AWARE_LAYOUT_REGIONS),
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    return converter.convert(str(pdf_path)).document.export_to_markdown()


def extract_text_hybrid(pdf_path):
    """Hybrid extraction: PyMuPDF4LLM first, fallback to Docling for scanned."""
    try:
        text = extract_text_pymupdf(pdf_path)
        word_count = len(text.split())
        
        if word_count < 50:
            log.info("Low text count (%d words), trying OCR: %s", word_count, pdf_path.name)
            try:
                ocr_text = extract_text_ocr(pdf_path)
                if len(ocr_text.split()) > word_count:
                    return clean_text(ocr_text)
            except Exception as e:
                log.warning("OCR failed: %s", e)
        
        return clean_text(text)
    except Exception as e:
        log.warning("PyMuPDF failed (%s), trying OCR: %s", e, pdf_path.name)
        return clean_text(extract_text_ocr(pdf_path))
