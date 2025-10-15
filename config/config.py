"""
Configuration file for Brain Tumor Classification Project
Author: [Your Name]
Date: [Current Date]
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

@dataclass
class ProjectConfig:
    """Main configuration class for the project"""
    
    # Project Information
    PROJECT_NAME: str = "Brain Tumor Classification with XAI"
    VERSION: str = "1.0.0"
    AUTHOR: str = "Your Name"
    DESCRIPTION: str = "Explainable AI for Brain Tumor Classification from MRI Images"
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    RESULTS_DIR: Path = BASE_DIR / "results"
    LOGS_DIR: Path = BASE_DIR / "results" / "logs"
    FIGURES_DIR: Path = BASE_DIR / "results" / "figures"
    
    # Data Configuration
    IMG_SIZE: Tuple[int, int] = (224, 224)
    IMG_CHANNELS: int = 3
    NUM_CLASSES: int = 4
    CLASS_NAMES: List[str] = None
    BATCH_SIZE: int = 32
    
    # Training Configuration
    EPOCHS: int = 50
    FINE_TUNE_EPOCHS: int = 20
    LEARNING_RATE: float = 0.001
    FINE_TUNE_LR: float = 0.0001
    VALIDATION_SPLIT: float = 0.2
    
    # Model Configuration
    MODEL_NAME: str = "brain_tumor_classifier"
    BACKBONE: str = "EfficientNetB0"  # Options: EfficientNetB0, VGG16, ResNet50V2
    DROPOUT_RATE: float = 0.3
    DENSE_UNITS: List[int] = None
    
    # Data Augmentation
    ROTATION_RANGE: int = 25
    WIDTH_SHIFT_RANGE: float = 0.2
    HEIGHT_SHIFT_RANGE: float = 0.2
    HORIZONTAL_FLIP: bool = True
    ZOOM_RANGE: float = 0.2
    BRIGHTNESS_RANGE: Tuple[float, float] = (0.8, 1.2)
    
    # Explainable AI Configuration
    LIME_NUM_FEATURES: int = 5
    LIME_NUM_SAMPLES: int = 1000
    GRADCAM_LAYER_NAME: str = "global_average_pooling2d"
    SHAP_NUM_SAMPLES: int = 100
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Hardware Configuration
    USE_GPU: bool = True
    MIXED_PRECISION: bool = True
    
    def __post_init__(self):
        """Initialize derived configurations"""
        if self.CLASS_NAMES is None:
            self.CLASS_NAMES = ['No Tumor', 'Glioma', 'Pituitary', 'Meningioma']
        
        if self.DENSE_UNITS is None:
            self.DENSE_UNITS = [512, 256]
        
        # Create directories
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.MODELS_DIR,
            self.RESULTS_DIR,
            self.LOGS_DIR,
            self.FIGURES_DIR,
            self.DATA_DIR / "train",
            self.DATA_DIR / "validation", 
            self.DATA_DIR / "test"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the class distribution for the dataset"""
        return {
            'No Tumor': 5000,
            'Glioma': 7000,
            'Pituitary': 7000,
            'Meningioma': 7000
        }
    
    def get_callbacks_config(self) -> Dict[str, Any]:
        """Get callbacks configuration"""
        return {
            'early_stopping': {
                'monitor': 'val_accuracy',
                'patience': 15,
                'restore_best_weights': True,
                'verbose': 1
            },
            'reduce_lr': {
                'monitor': 'val_loss',
                'factor': 0.5,
                'patience': 8,
                'min_lr': 1e-7,
                'verbose': 1
            },
            'model_checkpoint': {
                'filepath': str(self.MODELS_DIR / f'{self.MODEL_NAME}_best.h5'),
                'monitor': 'val_accuracy',
                'save_best_only': True,
                'save_weights_only': False,
                'verbose': 1
            },
            'csv_logger': {
                'filename': str(self.LOGS_DIR / 'training_log.csv'),
                'append': True
            },
            'tensorboard': {
                'log_dir': str(self.LOGS_DIR / 'tensorboard'),
                'histogram_freq': 1,
                'write_graph': True,
                'write_images': True
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }

# Global configuration instance
config = ProjectConfig()

# Environment-specific configurations
class DevelopmentConfig(ProjectConfig):
    """Development environment configuration"""
    BATCH_SIZE: int = 16
    EPOCHS: int = 10
    FINE_TUNE_EPOCHS: int = 5

class ProductionConfig(ProjectConfig):
    """Production environment configuration"""
    BATCH_SIZE: int = 64
    EPOCHS: int = 100
    FINE_TUNE_EPOCHS: int = 30
    MIXED_PRECISION: bool = True

class TestingConfig(ProjectConfig):
    """Testing environment configuration"""
    BATCH_SIZE: int = 8
    EPOCHS: int = 2
    FINE_TUNE_EPOCHS: int = 1
    IMG_SIZE: Tuple[int, int] = (64, 64)

def get_config(env: str = 'default') -> ProjectConfig:
    """Get configuration based on environment"""
    configs = {
        'development': DevelopmentConfig(),
        'production': ProductionConfig(),
        'testing': TestingConfig(),
        'default': ProjectConfig()
    }
    
    return configs.get(env.lower(), ProjectConfig())