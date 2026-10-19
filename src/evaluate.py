import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import CONFUSION_MATRIX_PATH, METRICS_PATH, REPORT_DIR


def evaluate_model(model, data_loader, device, class_names):
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    figure, axis = plt.subplots(figsize=(12, 10))
    display.plot(
        ax=axis,
        xticks_rotation=45,
        colorbar=False,
        values_format="d",
    )
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(figure)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }