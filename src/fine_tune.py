"""
Short fine-tune of the deployed LiteCNN checkpoint, aimed at the meningioma
over-prediction issue found by src/evaluate.py + src/calibrate_bias.py.

Runs on a stratified subsample of data/train (faster on CPU-only hardware),
validates on the untouched data/val split, and saves the best checkpoint to
finetuned_lite_model.pth WITHOUT touching best_lite_model.pth. Promote it only
after comparing against the baseline with src/evaluate.py.
"""

import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model2 import LiteCNN, FocalLoss, get_enhanced_transforms  # noqa: E402

SEED = 42
SAMPLES_PER_CLASS = 2000
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_WORKERS = 0  # this machine has ~6GB RAM; extra worker processes risk OOM

BASE_MODEL_PATH = PROJECT_ROOT / "artifacts" / "model" / "best_lite_model.pth"
OUT_MODEL_PATH = PROJECT_ROOT / "finetuned_lite_model.pth"
LOG_PATH = PROJECT_ROOT / "results" / "logs" / "finetune_log.txt"

CLASSES = [
    ("glioma", "Glioma"),
    ("meningioma", "Meningioma"),
    ("no_tumor", "no_tumor"),
    ("pituitary", "Pituitary"),
]


class SubsampledDataset(Dataset):
    def __init__(self, data_dir, transform, samples_per_class=None, seed=SEED):
        self.transform = transform
        self.images = []
        self.labels = []
        rng = random.Random(seed)
        for class_idx, (_, folder_name) in enumerate(CLASSES):
            class_dir = Path(data_dir) / folder_name
            files = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if samples_per_class is not None and len(files) > samples_per_class:
                files = rng.sample(files, samples_per_class)
            self.images.extend(files)
            self.labels.extend([class_idx] * len(files))
            print(f"  {folder_name}: {len(files)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), self.labels[idx]


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total


def log(f, msg):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w") as f:
        log(f, f"Loading baseline weights from {BASE_MODEL_PATH}")
        model = LiteCNN(num_classes=4)
        model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location="cpu"))

        train_transform, val_transform = get_enhanced_transforms()

        log(f, f"Building stratified train subsample ({SAMPLES_PER_CLASS}/class)...")
        train_ds = SubsampledDataset(
            PROJECT_ROOT / "data" / "train", train_transform, SAMPLES_PER_CLASS
        )
        log(f, "Loading validation set (data/val, full)...")
        val_ds = SubsampledDataset(PROJECT_ROOT / "data" / "val", val_transform)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        baseline_val_acc = evaluate(model, val_loader)
        log(f, f"Baseline val accuracy: {baseline_val_acc*100:.2f}%")

        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val_acc = baseline_val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

        log(f, f"\n{'Epoch':>5} {'TrainLoss':>10} {'TrainAcc':>9} {'ValAcc':>8} {'Time':>7}")
        log(f, "-" * 50)

        for epoch in range(EPOCHS):
            start = time.time()
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                _, pred = torch.max(output, 1)
                correct += (pred == target).sum().item()
                total += target.size(0)

            train_acc = correct / total
            val_acc = evaluate(model, val_loader)
            elapsed = time.time() - start

            log(
                f,
                f"{epoch+1:5d} {total_loss/len(train_loader):10.4f} "
                f"{train_acc*100:8.2f}% {val_acc*100:7.2f}% {elapsed:6.1f}s",
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                log(f, f"    -> new best (val acc {val_acc*100:.2f}%)")

        log(f, f"\nBest val accuracy: {best_val_acc*100:.2f}% (baseline was {baseline_val_acc*100:.2f}%)")

        if best_val_acc > baseline_val_acc:
            torch.save(best_state, OUT_MODEL_PATH)
            log(f, f"Saved improved weights to {OUT_MODEL_PATH}")
        else:
            log(f, "No improvement over baseline on validation set — not saving a new checkpoint.")


if __name__ == "__main__":
    main()
