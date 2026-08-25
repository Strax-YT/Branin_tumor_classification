"""
Evaluate the deployed LiteCNN checkpoint (best_lite_model.pth) on data/test.

Mirrors the exact preprocessing used by app/simple_direct_app.py so the
numbers reported here match what the Streamlit app actually does.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model2 import LiteCNN  # noqa: E402

# Class index order must match app/simple_direct_app.py's class_names list
# and model2.py's CONFIG['class_names'].
CLASSES = [
    ("Glioma Tumor", "Glioma"),
    ("Meningioma Tumor", "Meningioma"),
    ("No Tumor", "no_tumor"),
    ("Pituitary Tumor", "Pituitary"),
]

MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "best_lite_model.pth"
TEST_DIR = PROJECT_ROOT / "data" / "test"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_test_set():
    images, labels = [], []
    for class_idx, (_, folder_name) in enumerate(CLASSES):
        class_dir = TEST_DIR / folder_name
        files = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        images.extend(files)
        labels.extend([class_idx] * len(files))
        print(f"  {folder_name}: {len(files)} images")
    return images, labels


def main():
    print(f"Loading model from {MODEL_PATH}")
    model = LiteCNN(num_classes=4)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loading test set from {TEST_DIR}")
    image_paths, y_true = load_test_set()
    print(f"Total test images: {len(image_paths)}")

    y_pred = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensor = transform(img).unsqueeze(0)
            output = model(tensor)
            probs = F.softmax(output, dim=1)
            y_pred.append(int(torch.argmax(probs, dim=1).item()))

    class_labels = [name for name, _ in CLASSES]
    report = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True
    )
    report_text = classification_report(y_true, y_pred, target_names=class_labels)
    cm = confusion_matrix(y_true, y_pred)

    accuracy = report["accuracy"]
    print("\n" + "=" * 60)
    print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
    print("=" * 60)
    print(report_text)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOGS_DIR / "eval_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(LOGS_DIR / "classification_report.txt", "w") as f:
        f.write(f"Test accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(report_text)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_labels, yticklabels=class_labels,
    )
    plt.title(f"Confusion Matrix (Test Accuracy: {accuracy * 100:.2f}%)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    print(f"\nSaved: {LOGS_DIR / 'eval_metrics.json'}")
    print(f"Saved: {LOGS_DIR / 'classification_report.txt'}")
    print(f"Saved: {FIGURES_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
