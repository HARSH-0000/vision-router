import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification
from peft import LoraConfig, get_peft_model
from src.dataset import get_domain_dataloaders, DOMAINS

BASE_MODEL = "google/vit-base-patch16-224"

def train(domain: str, epochs: int = 2, lr: float = 3e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    train_ds, test_ds, labels = get_domain_dataloaders(domain, processor)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        ignore_mismatched_sizes=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Updated for newer transformers ViT implementation
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, lora_config).to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model(pixel_values=x, labels=y).loss
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(pixel_values=x).logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        print(f"[{domain}] Epoch {epoch+1} | Test Acc: {correct/total:.4f}")

    save_path = os.path.join("adapters", domain)
    model.save_pretrained(save_path)
    print(f"Saved adapter to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=list(DOMAINS.keys()), required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    train(args.domain, args.epochs)
