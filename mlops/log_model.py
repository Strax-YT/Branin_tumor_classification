"""
Retroactively logs the project's existing, real results into MLflow --
the currently-deployed baseline model, and the earlier fine-tune
experiment that was tried and correctly not promoted.

This does NOT train or change anything. It reads already-produced
files (artifacts/model/metadata.json, results/logs/*) and the model
via the same loader the API uses (app/direct_model_loader.py), then
logs them to a local file-based MLflow tracking store (./mlruns).

Run once with:  python mlops/log_model.py
Browse with:     mlflow ui   (from the project root)
"""

import json
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))
sys.path.insert(0, str(PROJECT_ROOT))

from direct_model_loader import load_direct_model  # noqa: E402
from src.fine_tune import EPOCHS, SAMPLES_PER_CLASS, BATCH_SIZE, LEARNING_RATE  # noqa: E402

EXPERIMENT_NAME = "neurosight-brain-tumor-classification"
REGISTERED_MODEL_NAME = "NeuroSight-LiteCNN"

# Real numbers transcribed from results/logs/finetune_comparison.txt --
# this file is also attached as a run artifact so the source is auditable.
FINETUNE_RESULTS = {
    "val_accuracy_before": 0.9100,
    "val_accuracy_after": 0.9225,
    "test_accuracy_before": 0.9162,
    "test_accuracy_after": 0.9098,
    "glioma_precision_before": 0.951, "glioma_precision_after": 0.88,
    "meningioma_precision_before": 0.803, "meningioma_precision_after": 0.85,
    "no_tumor_precision_before": 0.992, "no_tumor_precision_after": 1.00,
    "pituitary_precision_before": 0.936, "pituitary_precision_after": 0.92,
}


def log_baseline_run():
    metadata_path = PROJECT_ROOT / "artifacts" / "model" / "metadata.json"
    eval_metrics_path = PROJECT_ROOT / "results" / "logs" / "eval_metrics.json"
    classification_report_path = PROJECT_ROOT / "results" / "logs" / "classification_report.txt"
    confusion_matrix_path = PROJECT_ROOT / "results" / "figures" / "confusion_matrix.png"

    metadata = json.loads(metadata_path.read_text())
    eval_metrics = json.loads(eval_metrics_path.read_text())

    with mlflow.start_run(run_name="baseline-litecnn") as run:
        mlflow.log_params({
            "model_name": metadata["model_name"],
            "model_version": metadata["model_version"],
            "parameters": metadata["parameters"],
            "input_size": metadata["input_size"],
            "preprocessing": metadata["preprocessing"],
            "checkpoint_sha256": metadata["sha256"],
            "train_images": 27705,
            "val_images": 400,
            "test_images": 1098,
            "dataset_version_note": "DVC not active in this repo; image counts logged as a "
                                     "deterministic reproducibility anchor instead of a DVC hash.",
        })

        mlflow.log_metric("test_accuracy", eval_metrics["accuracy"])
        for class_name in metadata["classes"]:
            class_metrics = eval_metrics[class_name]
            slug = class_name.lower().replace(" ", "_")
            mlflow.log_metric(f"{slug}_precision", class_metrics["precision"])
            mlflow.log_metric(f"{slug}_recall", class_metrics["recall"])
            mlflow.log_metric(f"{slug}_f1", class_metrics["f1-score"])

        for artifact_path in [metadata_path, classification_report_path, confusion_matrix_path]:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        model = load_direct_model()
        mlflow.pytorch.log_model(model, "model")

        print(f"Logged baseline run: {run.info.run_id}")
        return run.info.run_id


def register_baseline(run_id):
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", result.version)
    print(f"Registered {REGISTERED_MODEL_NAME} v{result.version}, alias 'champion' set.")
    return result.version


def log_finetune_run():
    log_path = PROJECT_ROOT / "results" / "logs" / "finetune_log.txt"
    comparison_path = PROJECT_ROOT / "results" / "logs" / "finetune_comparison.txt"

    with mlflow.start_run(run_name="finetune-experiment-not-promoted") as run:
        mlflow.set_tag("outcome", "not_promoted")
        mlflow.log_params({
            "base_checkpoint": "best_lite_model.pth",
            "epochs": EPOCHS,
            "samples_per_class": SAMPLES_PER_CLASS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        })
        mlflow.log_metrics(FINETUNE_RESULTS)

        for artifact_path in [comparison_path, log_path]:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        print(f"Logged fine-tune run: {run.info.run_id} (not registered -- not promoted)")


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    baseline_run_id = log_baseline_run()
    register_baseline(baseline_run_id)
    log_finetune_run()

    print("\nDone. Browse results with: mlflow ui")


if __name__ == "__main__":
    main()
