from pypdf import PdfReader
from docx import Document

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(file):
    name = file.name.lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(file)

    elif name.endswith(".docx"):
        return extract_text_from_docx(file)

    elif name.endswith(".txt"):
        return str(file.read(), "utf-8")

    return ""
