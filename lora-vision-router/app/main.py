"""FastAPI inference service for the multi-adapter ViT."""

import io
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from peft import PeftModel
from transformers import AutoImageProcessor, ViTForImageClassification

from src.dataset import BASE_MODEL, CIFAR10_CLASSES, DOMAINS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ADAPTER_ROOT = Path("adapters")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="Multi-Adapter LoRA Vision Router",
    description="Image classification with domain-specific LoRA adapters.",
    version="1.1.0",
)

# Adapter selection mutates shared model state. Serialize adapter selection and
# inference so simultaneous requests cannot run with another request's adapter.
_model_load_lock = threading.Lock()
_inference_lock = threading.RLock()


def _load_runtime():
    with _model_load_lock:
        if hasattr(app.state, "model") and hasattr(app.state, "processor"):
            return app.state.model, app.state.processor

        adapter_paths = {
            domain: ADAPTER_ROOT / domain for domain in DOMAINS
        }
        missing = [
            str(path / "adapter_config.json")
            for path in adapter_paths.values()
            if not (path / "adapter_config.json").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Trained adapters are missing. Train both domains first. "
                "Missing: " + ", ".join(missing)
            )

        processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
        base_model = ViTForImageClassification.from_pretrained(
            BASE_MODEL,
            num_labels=len(CIFAR10_CLASSES),
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_paths["vehicles"]),
            adapter_name="vehicles",
        )
        model.load_adapter(str(adapter_paths["animals"]), adapter_name="animals")
        model.to(DEVICE)
        model.eval()

        app.state.model = model
        app.state.processor = processor
        return model, processor


def _run_prediction(image: Image.Image, domain: str) -> dict:
    model, processor = _load_runtime()
    config = DOMAINS[domain]

    with _inference_lock, torch.inference_mode():
        model.set_adapter(domain)
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        logits = model(pixel_values=pixel_values).logits.squeeze(0)

        allowed_ids = list(config["original_ids"])
        allowed_logits = logits[allowed_ids]
        probabilities = torch.softmax(allowed_logits, dim=0)
        domain_prediction = int(probabilities.argmax().item())
        predicted_class_id = allowed_ids[domain_prediction]

        return {
            "domain": domain,
            "prediction": config["labels"][domain_prediction],
            "class_id": predicted_class_id,
            "confidence": round(float(probabilities[domain_prediction].item()), 4),
        }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    domain: str = Query(..., description="Adapter domain: vehicles or animals"),
    file: UploadFile = File(...),
):
    if domain not in DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain {domain!r}. Choose from: {', '.join(DOMAINS)}",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload must be an image.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")

    try:
        with Image.open(io.BytesIO(contents)) as uploaded_image:
            image = uploaded_image.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    started_at = time.perf_counter()
    try:
        result = await run_in_threadpool(_run_prediction, image, domain)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Model initialization or inference failed: {error}",
        )

    result["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
    return result
