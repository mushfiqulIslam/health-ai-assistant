import os
import re
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OCR_SPACE_URL = "https://api.ocr.space/parse/image"
DEFAULT_SAMPLE_TEXT = """Patient: John Doe
Rx
Amoxicillin 500 mg
Take 1 capsule by mouth three times daily for 7 days
After meals
Refill: 0
Doctor: Dr. Rahman"""

load_dotenv(BASE_DIR / ".env")


class PrescriptionField(BaseModel):
    label: str
    value: str


class PrescriptionResponse(BaseModel):
    provider: str
    raw_text: str
    parsed_fields: list[PrescriptionField]


app = FastAPI(title="Prescription OCR Demo", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/prescription/read", response_model=PrescriptionResponse)
async def read_prescription(
    image: UploadFile = File(...),
    provider: Literal["mock", "ocr_space"] = Form("mock"),
) -> PrescriptionResponse:
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    if provider == "mock":
        raw_text = DEFAULT_SAMPLE_TEXT
    else:
        raw_text = await extract_text_with_ocr_space(image_bytes, image.filename or "prescription.png")

    return PrescriptionResponse(
        provider=provider,
        raw_text=raw_text,
        parsed_fields=parse_prescription_text(raw_text),
    )


async def extract_text_with_ocr_space(image_bytes: bytes, filename: str) -> str:
    api_key = os.getenv("OCR_SPACE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OCR_SPACE_API_KEY is not configured. Use mock mode or set the API key.",
        )

    files = {"file": (filename, image_bytes, "application/octet-stream")}
    data = {
        "language": "eng",
        "isOverlayRequired": "false",
        "OCREngine": "2",
        "scale": "true",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            OCR_SPACE_URL,
            headers={"apikey": api_key},
            data=data,
            files=files,
        )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OCR provider returned HTTP {response.status_code}.")

    payload = response.json()
    if payload.get("IsErroredOnProcessing"):
        message = payload.get("ErrorMessage") or ["OCR processing failed."]
        raise HTTPException(status_code=502, detail=", ".join(message))

    parsed_text = "\n".join(
        item.get("ParsedText", "").strip()
        for item in payload.get("ParsedResults", [])
        if item.get("ParsedText")
    ).strip()

    if not parsed_text:
        raise HTTPException(status_code=422, detail="No text detected in the uploaded image.")

    return parsed_text


def parse_prescription_text(text: str) -> list[PrescriptionField]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def find_line(pattern: str) -> str:
        return next((line for line in lines if re.search(pattern, line, re.IGNORECASE)), "Not detected")

    medicine_line = find_line(r"(mg|ml|tablet|capsule|syrup|ointment|injection)")
    dosage_line = find_line(r"(\d+\s?(mg|ml)|\btablet\b|\bcapsule\b|\bteaspoon\b)")
    frequency_line = find_line(r"(daily|times daily|once|twice|every|after meals|before meals|at night)")
    duration_line = find_line(r"(for\s+\d+\s+(day|days|week|weeks)|x\s*\d+\s*days)")
    doctor_line = find_line(r"(doctor|dr\.)")

    if medicine_line == "Not detected" and lines:
        medicine_line = lines[0]

    return [
        PrescriptionField(label="Likely medicine", value=medicine_line),
        PrescriptionField(label="Likely dosage", value=dosage_line),
        PrescriptionField(label="Likely frequency", value=frequency_line),
        PrescriptionField(label="Likely duration", value=duration_line),
        PrescriptionField(label="Doctor / issuer", value=doctor_line),
    ]
