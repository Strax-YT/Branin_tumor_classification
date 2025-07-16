import mlflow
import mlflow.tensorflow
import tensorflow as tf

with mlflow.start_run():
    # Your preprocessing and model building
    # Train model
    mlflow.log_param("model", "ResNet50")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.tensorflow.log_model(model, "model")
