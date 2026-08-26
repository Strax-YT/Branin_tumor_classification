import io

import torch
from PIL import Image
from torchvision import transforms

from direct_model_loader import load_direct_model, ExplainabilityWrapper

CLASS_NAMES = ["Glioma Tumor", "Meningioma Tumor", "No Tumor", "Pituitary Tumor"]

_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def test_attention_map_is_produced(sample_image_bytes):
    model = load_direct_model()
    explainer = ExplainabilityWrapper(model, CLASS_NAMES)

    image = Image.open(io.BytesIO(sample_image_bytes("glioma"))).convert("RGB")
    input_tensor = _transform(image).unsqueeze(0)

    result = explainer.predict_with_explanations(input_tensor)
    assert result["attention_map"] is not None


def test_overlay_matches_original_image_dimensions(sample_image_bytes):
    model = load_direct_model()
    explainer = ExplainabilityWrapper(model, CLASS_NAMES)

    image = Image.open(io.BytesIO(sample_image_bytes("glioma"))).convert("RGB")
    input_tensor = _transform(image).unsqueeze(0)

    result = explainer.predict_with_explanations(input_tensor)
    overlay = explainer.create_attention_overlay(image, result["attention_map"])

    width, height = image.size
    assert overlay.shape == (height, width, 3)


def test_predicted_class_is_within_range(sample_image_bytes):
    model = load_direct_model()
    explainer = ExplainabilityWrapper(model, CLASS_NAMES)

    image = Image.open(io.BytesIO(sample_image_bytes("meningioma"))).convert("RGB")
    input_tensor = _transform(image).unsqueeze(0)

    result = explainer.predict_with_explanations(input_tensor)
    assert 0 <= result["predicted_class"] < len(CLASS_NAMES)
