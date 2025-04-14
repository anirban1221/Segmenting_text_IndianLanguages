from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import TextSegmentModel
from segment import extract_text_from_pdf, segment_text

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Update this if your frontend runs on a different port
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your text segmentation model
model = TextSegmentModel("your_model.pt")

class SegmentRequest(BaseModel):
    text: str
    language: str

@app.post("/segment-text")
async def segment_text_endpoint(payload: SegmentRequest):
    segments = segment_text(payload.text)
    embeddings = [model.get_embeddings(seg).tolist() for seg in segments]
    return {"segments": embeddings}

@app.post("/segment-pdf")
async def segment_pdf_endpoint(file: UploadFile = File(...)):
    try:
        text = extract_text_from_pdf(file.file)
        segments = segment_text(text)
        embeddings = [model.get_embeddings(seg).tolist() for seg in segments]
        return {"segments": embeddings}
    except Exception as e:
        # Log the error here if needed
        return {"error": str(e)}