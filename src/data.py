
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    IMAGE_SIZE,
    MEAN,
    NUM_WORKERS,
    RANDOM_SEED,
    STD,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


def get_transforms():
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    return train_transform, evaluation_transform


def load_datasets():
    train_transform, evaluation_transform = get_transforms()

    # Download the RGB EuroSAT dataset on the first run.
    base_dataset = datasets.EuroSAT(
        root=str(DATA_DIR),
        download=True,
        transform=None,
    )

    labels = base_dataset.targets
    all_indices = list(range(len(labels)))

    train_indices, remaining_indices = train_test_split(
        all_indices,
        test_size=(1.0 - TRAIN_RATIO),
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    remaining_labels = [labels[i] for i in remaining_indices]
    relative_test_ratio = TEST_RATIO / (
        VALIDATION_RATIO + TEST_RATIO
    )

    validation_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=relative_test_ratio,
        random_state=RANDOM_SEED,
        stratify=remaining_labels,
    )

    # Separate dataset instances allow different transforms
    # for training and evaluation.
    train_dataset = datasets.EuroSAT(
        root=str(DATA_DIR),
        download=False,
        transform=train_transform,
    )

    evaluation_dataset = datasets.EuroSAT(
        root=str(DATA_DIR),
        download=False,
        transform=evaluation_transform,
    )

    train_subset = Subset(train_dataset, train_indices)
    validation_subset = Subset(evaluation_dataset, validation_indices)
    test_subset = Subset(evaluation_dataset, test_indices)

    return train_subset, validation_subset, test_subset, base_dataset.classes


def create_dataloaders():
    train_dataset, validation_dataset, test_dataset, class_names = (
        load_datasets()
    )

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, validation_loader, test_loader, class_names
