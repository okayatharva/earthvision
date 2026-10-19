import random

import numpy as np
import torch
from torch import nn

from src.config import (
    EPOCHS,
    LEARNING_RATE,
    MEAN,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    STD,
    IMAGE_SIZE,
)
from src.data import create_dataloaders
from src.evaluate import evaluate_model
from src.model import build_model, get_device


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, data_loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += batch_size

    return total_loss / total, correct / total


def main():
    set_seed()

    device = get_device()
    print(f"Using device: {device}")

    train_loader, validation_loader, test_loader, class_names = (
        create_dataloaders()
    )

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(validation_loader.dataset)}")
    print(f"Test images: {len(test_loader.dataset)}")

    model = build_model(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "mean": list(MEAN),
                "std": list(STD),
            }
            torch.save(checkpoint, MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH}")

    print("\nLoading best checkpoint for final test evaluation.")
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        class_names,
    )

    print("\nFinal test metrics:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nSaved evaluation report to reports/metrics.json")
    print("Saved confusion matrix to reports/confusion_matrix.png")


if __name__ == "__main__":
    main()