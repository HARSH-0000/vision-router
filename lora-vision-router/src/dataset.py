"""CIFAR-10 domain datasets for the vision router."""

from pathlib import Path
from typing import Any

from torch.utils.data import Subset
from torchvision import datasets, transforms

BASE_MODEL = "google/vit-base-patch16-224"

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

DOMAINS = {
    "vehicles": {
        "original_ids": (0, 1, 8, 9),
        "labels": ("airplane", "automobile", "ship", "truck"),
    },
    "animals": {
        "original_ids": (2, 3, 4, 5, 6, 7),
        "labels": ("bird", "cat", "deer", "dog", "frog", "horse"),
    },
}


def get_domain_indices(targets: list[int], domain: str) -> list[int]:
    """Return dataset indices belonging to a configured domain."""
    if domain not in DOMAINS:
        raise ValueError(
            f"Unknown domain {domain!r}. Choose from: {', '.join(DOMAINS)}"
        )

    allowed_ids = set(DOMAINS[domain]["original_ids"])
    return [index for index, target in enumerate(targets) if target in allowed_ids]


def get_domain_datasets(
    domain: str,
    processor: Any,
    root: str | Path = "./data",
    download: bool = True,
) -> tuple[Subset, Subset]:
    """Load CIFAR-10 train/test subsets without changing their original labels.

    Both adapters use a 10-class classifier. Keeping CIFAR-10's original labels
    makes the classifier dimensions identical across domains and avoids
    incompatible adapter-specific classifier shapes.
    """
    if domain not in DOMAINS:
        raise ValueError(
            f"Unknown domain {domain!r}. Choose from: {', '.join(DOMAINS)}"
        )

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=processor.image_mean,
                std=processor.image_std,
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=str(root), train=True, download=download, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root=str(root), train=False, download=download, transform=transform
    )

    train_indices = get_domain_indices(train_dataset.targets, domain)
    test_indices = get_domain_indices(test_dataset.targets, domain)

    return (
        Subset(train_dataset, train_indices),
        Subset(test_dataset, test_indices),
    )
