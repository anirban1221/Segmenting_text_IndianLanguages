import fitz 

def extract_text_from_pdf(file) -> str:
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"
    finally:
        doc.close()  

def segment_text(text: str) -> list[str]: 
    return [sentence.strip() for sentence in text.split('.') if sentence.strip()]