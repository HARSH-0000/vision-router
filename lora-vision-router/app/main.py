import io
import time
import torch
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from transformers import AutoImageProcessor, ViTForImageClassification
from peft import PeftModel
from src.dataset import DOMAINS

BASE_MODEL = "google/vit-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Multi-Adapter LoRA Vision Router")

processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
base_model = ViTForImageClassification.from_pretrained(BASE_MODEL)

model = PeftModel.from_pretrained(base_model, "adapters/vehicles", adapter_name="vehicles")
model.load_adapter("adapters/animals", adapter_name="animals")
model.to(DEVICE).eval()

@app.post("/predict")
async def predict(
    domain: str = Query(..., description="Adapter domain: vehicles or animals"),
    file: UploadFile = File(...),
):
    if domain not in DOMAINS:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")

    start = time.perf_counter()
    model.set_adapter(domain)

    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred_idx = probs.argmax().item()

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return {
        "domain": domain,
        "prediction": DOMAINS[domain]["labels"][pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "latency_ms": latency_ms,
    }
