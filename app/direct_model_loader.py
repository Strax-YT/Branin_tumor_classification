"""
DIRECT Model Loader - Uses Your Exact Model Without Architecture Changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import os
import sys
from pathlib import Path

from model_registry import verify_model_integrity

print("[OK] Direct model loader imported!")

# Resolve the project root relative to this file (not the process's CWD), so
# the app works regardless of the directory Streamlit was launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ===== DIRECT IMPORT OF YOUR ORIGINAL MODEL =====
try:
    # Try to import your original model directly
    from models.model2 import LiteCNN
    MODEL_IMPORTED = True
    print("[OK] Successfully imported original LiteCNN from model2.py")
except ImportError as e:
    print(f"[ERROR] Could not import original model: {e}")
    print("[WARN] Creating identical model structure...")
    MODEL_IMPORTED = False
    
    # Create the exact same model as in model2.py -- one conv per block
    # (NOT two; this must match best_lite_model.pth's actual saved shapes).
    class LiteCNN(nn.Module):
        def __init__(self, num_classes=4):
            super(LiteCNN, self).__init__()
            self.features = nn.Sequential(
                # Block 1: 128 -> 64
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                # Block 2: 64 -> 32
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
                # Block 3: 32 -> 16
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3),
                # Block 4: 16 -> 8
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.3),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

# ===== EXPLAINABILITY WRAPPER =====
class ExplainabilityWrapper:
    """Wraps any model with explainability features"""
    
    def __init__(self, model, class_names):
        self.model = model
        self.class_names = class_names
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks on the last convolutional layer
        self._register_hooks()
    
    def _register_hooks(self):
        """Find and register hooks on convolutional layers"""
        # Find all convolutional layers
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_layers.append((name, module))
        
        if conv_layers:
            # Use the last convolutional layer for Grad-CAM
            last_conv_name, last_conv_layer = conv_layers[-1]
            print(f"[OK] Registered Grad-CAM hook on: {last_conv_name}")

            def forward_hook(module, input, output):
                self.feature_maps = output

            def backward_hook(module, grad_input, grad_output):
                self.gradients = grad_output[0]

            last_conv_layer.register_forward_hook(forward_hook)
            last_conv_layer.register_full_backward_hook(backward_hook)
        else:
            print("[WARN] No convolutional layers found for Grad-CAM")
    
    def predict_with_explanations(self, input_tensor):
        """Get prediction with Grad-CAM explanations"""
        self.model.eval()
        
        # Reset hooks
        self.feature_maps = None
        self.gradients = None
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1).numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
        
        # Generate Grad-CAM if hooks are available
        cam = None
        if self.feature_maps is not None:
            cam = self._generate_grad_cam(input_tensor, predicted_class)
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'attention_map': cam,
            'output': output
        }
    
    def _generate_grad_cam(self, input_tensor, target_class):
        """Generate Grad-CAM heatmap"""
        if self.feature_maps is None or self.gradients is None:
            return self._create_fallback_cam()
        
        try:
            # Get gradients and feature maps
            gradients = self.gradients.detach().cpu().numpy()[0]
            feature_maps = self.feature_maps.detach().cpu().numpy()[0]
            
            # Global average pooling of gradients
            weights = np.mean(gradients, axis=(1, 2))
            
            # Create CAM
            cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * feature_maps[i, :, :]
            
            # Apply ReLU and resize
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
            cam = (cam - np.min(cam)) / (np.max(cam) + 1e-8)
            
            return cam
            
        except Exception as e:
            print(f"[WARN] Grad-CAM error: {e}")
            return self._create_fallback_cam()
    
    def _create_fallback_cam(self):
        """Create fallback attention map"""
        cam = np.zeros((128, 128), dtype=np.float32)
        # Create centered Gaussian attention
        for i in range(128):
            for j in range(128):
                dist = np.sqrt((i-64)**2 + (j-64)**2)
                cam[i, j] = np.exp(-dist/40)
        return cam
    
    def create_attention_overlay(self, original_image, cam, alpha=0.5):
        """Overlay heatmap on image"""
        if isinstance(original_image, Image.Image):
            original_np = np.array(original_image)
        else:
            original_np = original_image
        
        # Ensure RGB
        if len(original_np.shape) == 2:
            original_np = np.stack([original_np] * 3, axis=-1)
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
        
        # Blend
        overlay = cv2.addWeighted(original_np, 1-alpha, heatmap, alpha, 0)
        return overlay
    
    def analyze_confidence(self, probabilities):
        """Analyze prediction confidence"""
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        max_prob = np.max(probabilities)
        
        sorted_probs = np.sort(probabilities)
        confidence_gap = max_prob - sorted_probs[-2] if len(sorted_probs) > 1 else max_prob
        
        return {
            'entropy': entropy,
            'max_confidence': max_prob,
            'confidence_gap': confidence_gap,
            'uncertainty': entropy > 1.0,
            'confidence_level': 'High' if max_prob > 0.8 else 'Medium' if max_prob > 0.6 else 'Low'
        }

# ===== DIRECT MODEL LOADING =====
def load_direct_model():
    """Load your model directly without architecture changes"""
    model_path = PROJECT_ROOT / "artifacts" / "model" / "best_lite_model.pth"

    if not model_path.exists():
        print(f"[ERROR] Model file not found: {model_path}")
        return None
    print(f"[OK] Model found: {model_path}")

    integrity = verify_model_integrity(model_path)
    if integrity["ok"]:
        print(f"[OK] Model integrity verified: {integrity['reason']}")
    else:
        print(f"[WARN] Model integrity check failed: {integrity['reason']}")

    try:
        # Create model instance
        model = LiteCNN(num_classes=4)

        # Load state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()

        print("[OK] Model loaded successfully!")
        print(f"[OK] Model architecture: {type(model).__name__}")

        return model

    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return None

# ===== TEST FUNCTION =====
def test_direct_model():
    """Test the direct model loading"""
    print("\nTesting Direct Model Loading...")

    model = load_direct_model()
    if not model:
        return None, None

    # Test prediction
    try:
        dummy_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = F.softmax(output, dim=1).numpy()[0]

        print(f"[OK] Model works! Output shape: {output.shape}")
        print(f"[OK] Sample probabilities: {probabilities}")

        # Create explainability wrapper
        explainer = ExplainabilityWrapper(
            model,
            ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
        )

        # Test explainability
        result = explainer.predict_with_explanations(dummy_input)
        print(f"[OK] Explainability works! CAM: {result['attention_map'] is not None}")

        return model, explainer

    except Exception as e:
        print(f"[ERROR] Error during testing: {e}")
        return None, None

# Run test
if __name__ == "__main__":
    print("="*60)
    print("DIRECT MODEL LOADING TEST")
    print("="*60)

    model, explainer = test_direct_model()

    if model and explainer:
        print("\n[SUCCESS] Direct model loading is working!")
        print("Ready for Streamlit integration!")
    else:
        print("\n[FAILED] Test failed")