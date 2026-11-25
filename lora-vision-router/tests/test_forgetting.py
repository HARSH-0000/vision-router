"""Integration test for adapter isolation.

This test runs only after both adapters have been trained and saved. It verifies
that loading the second adapter does not change predictions from the first.
"""

from pathlib import Path

import pytest
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification

from src.dataset import BASE_MODEL, CIFAR10_CLASSES, DOMAINS, get_domain_datasets

ADAPTER_ROOT = Path("adapters")
pytestmark = pytest.mark.integration


def evaluate_accuracy(model, domain, processor, device):
    _, test_dataset = get_domain_datasets(domain, processor)
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    allowed_ids = list(DOMAINS[domain]["original_ids"])
    correct = 0
    total = 0

    model.set_adapter(domain)
    model.eval()
    with torch.inference_mode():
        for pixel_values, labels in loader:
            logits = model(pixel_values=pixel_values.to(device)).logits
            predictions = logits[:, allowed_ids].argmax(dim=1)
            predicted_original_ids = torch.tensor(
                [allowed_ids[index] for index in predictions.cpu().tolist()]
            )
            correct += (predicted_original_ids == labels).sum().item()
            total += labels.size(0)

    return correct / total if total else 0.0


def test_loading_second_adapter_does_not_change_first_adapter():
    required_files = [
        ADAPTER_ROOT / domain / "adapter_config.json" for domain in DOMAINS
    ]
    if not all(path.is_file() for path in required_files):
        pytest.skip("Train both adapters before running this integration test.")

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = ViTForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CIFAR10_CLASSES),
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(
        base_model, str(ADAPTER_ROOT / "vehicles"), adapter_name="vehicles"
    ).to(device)

    vehicles_before = evaluate_accuracy(model, "vehicles", processor, device)

    model.load_adapter(str(ADAPTER_ROOT / "animals"), adapter_name="animals")
    vehicles_after = evaluate_accuracy(model, "vehicles", processor, device)
    animals_accuracy = evaluate_accuracy(model, "animals", processor, device)

    assert abs(vehicles_before - vehicles_after) < 1e-8, (
        "Loading the animals adapter changed vehicle adapter accuracy: "
        f"{vehicles_before:.4f} -> {vehicles_after:.4f}"
    )
    assert 0.0 <= animals_accuracy <= 1.0
