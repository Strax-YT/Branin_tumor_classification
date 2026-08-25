"""
Try to fix the meningioma over-prediction problem (see results/figures/confusion_matrix.png)
with a per-class logit bias, tuned on data/val (never touched by evaluate.py) and then
verified on data/test to confirm it's a genuine improvement and not overfitting to one split.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model2 import LiteCNN  # noqa: E402

CLASSES = [
    ("Glioma Tumor", "Glioma"),
    ("Meningioma Tumor", "Meningioma"),
    ("No Tumor", "no_tumor"),
    ("Pituitary Tumor", "Pituitary"),
]
MENINGIOMA_IDX = 1

MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "best_lite_model.pth"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_logits(model, split_dir):
    logits, labels = [], []
    for class_idx, (_, folder_name) in enumerate(CLASSES):
        class_dir = split_dir / folder_name
        files = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        for path in files:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)
            logits.append(output.squeeze(0).numpy())
            labels.append(class_idx)
    return np.array(logits), np.array(labels)


def macro_f1_with_bias(logits, labels, delta):
    biased = logits.copy()
    biased[:, MENINGIOMA_IDX] -= delta
    preds = np.argmax(biased, axis=1)
    return f1_score(labels, preds, average="macro")


def main():
    model = LiteCNN(num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print("Computing logits on validation set (data/val)...")
    val_logits, val_labels = get_logits(model, PROJECT_ROOT / "data" / "val")

    print("Sweeping meningioma logit bias on validation set...")
    deltas = np.arange(0.0, 3.05, 0.1)
    scores = [macro_f1_with_bias(val_logits, val_labels, d) for d in deltas]
    best_delta = float(deltas[int(np.argmax(scores))])
    print(f"Best delta on val: {best_delta:.2f} (macro-F1 {max(scores):.4f} vs "
          f"{scores[0]:.4f} at delta=0)")

    print("\nComputing logits on test set (data/test) to verify...")
    test_logits, test_labels = get_logits(model, PROJECT_ROOT / "data" / "test")

    class_labels = [name for name, _ in CLASSES]

    for label, delta in [("Baseline (delta=0)", 0.0), (f"Calibrated (delta={best_delta:.2f})", best_delta)]:
        biased = test_logits.copy()
        biased[:, MENINGIOMA_IDX] -= delta
        preds = np.argmax(biased, axis=1)
        acc = (preds == test_labels).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds, labels=list(range(4))
        )
        print(f"\n{label} -- Test accuracy: {acc*100:.2f}%")
        for i, name in enumerate(class_labels):
            print(f"  {name:20s} precision={precision[i]:.3f} recall={recall[i]:.3f} f1={f1[i]:.3f}")


if __name__ == "__main__":
    main()
