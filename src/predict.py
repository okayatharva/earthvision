import torch
from PIL import Image
from torchvision import transforms

from src.config import MODEL_PATH
from src.model import build_model, get_device


def load_model(model_path=MODEL_PATH):
    device = get_device()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Train the model first using: python -m src.train"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_transform = transforms.Compose(
        [
            transforms.Resize(
                (checkpoint["image_size"], checkpoint["image_size"])
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                tuple(checkpoint["mean"]),
                tuple(checkpoint["std"]),
            ),
        ]
    )

    return model, class_names, image_transform, device


def predict_image(
    image: Image.Image,
    model,
    class_names,
    image_transform,
    device,
):
    image = image.convert("RGB")
    tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_count = min(3, len(class_names))
    scores, indices = torch.topk(probabilities, k=top_count)

    results = [
        {
            "class": class_names[index.item()],
            "probability": score.item(),
        }
        for score, index in zip(scores, indices)
    ]

    return results