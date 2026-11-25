"""Train one domain-specific LoRA adapter for the shared 10-class ViT."""

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification

from src.dataset import BASE_MODEL, CIFAR10_CLASSES, DOMAINS, get_domain_datasets

BATCH_SIZE = 64
LORA_TARGET_MODULES = ["query", "value"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader: DataLoader, domain: str, device: torch.device) -> float:
    allowed_ids = torch.tensor(
        DOMAINS[domain]["original_ids"], device=device, dtype=torch.long
    )
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            logits = model(pixel_values=pixel_values).logits
            domain_logits = logits.index_select(dim=1, index=allowed_ids)
            domain_labels = torch.tensor(
                [
                    DOMAINS[domain]["original_ids"].index(int(label))
                    for label in labels.tolist()
                ],
                device=device,
            )
            predictions = domain_logits.argmax(dim=1)
            correct += (predictions == domain_labels).sum().item()
            total += labels.size(0)

    return correct / total if total else 0.0


def train(
    domain: str,
    epochs: int = 2,
    learning_rate: float = 3e-4,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
    data_dir: str = "./data",
    output_dir: str = "adapters",
) -> float:
    if domain not in DOMAINS:
        raise ValueError(f"Unsupported domain: {domain}")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    train_dataset, test_dataset = get_domain_datasets(
        domain, processor, root=data_dir
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Both domain adapters must have the same classifier shape so they can be
    # loaded into one shared PEFT model. CIFAR-10's original 0-9 labels are kept.
    base_model = ViTForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(CIFAR10_CLASSES),
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base_model, lora_config).to(device)
    model.print_trainable_parameters()

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        batches = 0

        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = model(pixel_values=pixel_values, labels=labels).loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        accuracy = evaluate(model, test_loader, domain, device)
        average_loss = running_loss / max(batches, 1)
        print(
            f"[{domain}] epoch {epoch + 1}/{epochs} "
            f"loss={average_loss:.4f} domain_accuracy={accuracy:.4f}"
        )

    adapter_path = Path(output_dir) / domain
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)

    metadata = {
        "domain": domain,
        "base_model": BASE_MODEL,
        "num_labels": len(CIFAR10_CLASSES),
        "class_ids": list(DOMAINS[domain]["original_ids"]),
        "labels": list(DOMAINS[domain]["labels"]),
        "epochs": epochs,
        "seed": seed,
    }
    (adapter_path / "domain_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Saved {domain} adapter to {adapter_path}")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=tuple(DOMAINS), required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="adapters")
    args = parser.parse_args()

    train(
        domain=args.domain,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
