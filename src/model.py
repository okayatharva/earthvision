
import torch
from torch import nn

from src.config import IMAGE_SIZE


class EarthVisionCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)


def build_model(num_classes: int) -> EarthVisionCNN:
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")

    return EarthVisionCNN(num_classes=num_classes)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


if __name__ == "__main__":
    model = build_model(num_classes=10)
    sample = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    output = model(sample)

    print("Model:", model.__class__.__name__)
    print("Input shape:", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))
    print("Device available:", get_device())
