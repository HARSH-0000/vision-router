from torch.utils.data import Subset
from torchvision import datasets, transforms

DOMAINS = {
    "vehicles": {
        "original_ids": [0, 1, 8, 9],
        "labels": ["airplane", "automobile", "ship", "truck"],
    },
    "animals": {
        "original_ids": [2, 3, 4, 5, 6, 7],
        "labels": ["bird", "cat", "deer", "dog", "frog", "horse"],
    },
}

class RemappedSubset(Subset):
    def __init__(self, dataset, indices, id_map):
        super().__init__(dataset, indices)
        self.id_map = id_map

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return img, self.id_map[target]

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]


def get_domain_dataloaders(domain: str, processor, batch_size: int = 64):
    cfg = DOMAINS[domain]
    id_map = {orig: new for new, orig in enumerate(cfg["original_ids"])}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    train_raw = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_raw = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_idx = [i for i, y in enumerate(train_raw.targets) if y in id_map]
    test_idx = [i for i, y in enumerate(test_raw.targets) if y in id_map]

    return (
        RemappedSubset(train_raw, train_idx, id_map),
        RemappedSubset(test_raw, test_idx, id_map),
        cfg["labels"],
    )
