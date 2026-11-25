import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification
from peft import PeftModel
from src.dataset import get_domain_dataloaders

BASE_MODEL = "google/vit-base-patch16-224"

def evaluate_adapter(model, domain, processor):
    _, test_ds, _ = get_domain_dataloaders(domain, processor)
    loader = DataLoader(test_ds, batch_size=64)
    model.set_adapter(domain)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(pixel_values=x).logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def test_zero_catastrophic_forgetting():
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    base = ViTForImageClassification.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, "adapters/vehicles", adapter_name="vehicles")
    model.load_adapter("adapters/animals", adapter_name="animals")

    vehicles_acc = evaluate_adapter(model, "vehicles", processor)
    animals_acc = evaluate_adapter(model, "animals", processor)

    assert vehicles_acc > 0.90, f"Vehicles accuracy regressed: {vehicles_acc:.2f}"
    assert animals_acc > 0.85, f"Animals accuracy regressed: {animals_acc:.2f}"
