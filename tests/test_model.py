
import pytest
import torch

from src.model import build_model


def test_model_output_shape():
    model = build_model(num_classes=10)
    images = torch.randn(4, 3, 64, 64)

    outputs = model(images)

    assert outputs.shape == (4, 10)


def test_model_output_is_finite():
    model = build_model(num_classes=10)
    images = torch.randn(2, 3, 64, 64)

    outputs = model(images)

    assert torch.isfinite(outputs).all()


def test_model_supports_different_class_counts():
    model = build_model(num_classes=5)
    images = torch.randn(2, 3, 64, 64)

    outputs = model(images)

    assert outputs.shape == (2, 5)


def test_invalid_class_count_raises_error():
    with pytest.raises(ValueError):
        build_model(num_classes=0)
