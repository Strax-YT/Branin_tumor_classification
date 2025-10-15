import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
import warnings
import gc
warnings.filterwarnings('ignore')


# Paths

BASE_DATA_PATH = r"D:\Branin_tumor_classification\data"
TRAIN_DATA_PATH = os.path.join(BASE_DATA_PATH, "train")
VAL_DATA_PATH = os.path.join(BASE_DATA_PATH, "val")
TEST_DATA_PATH = os.path.join(BASE_DATA_PATH, "test")

CONFIG = {
    'img_size': 128,
    'batch_size': 16,
    'epochs': 60,          
    'learning_rate': 0.001,
    'num_classes': 4,
    'class_names': ['glioma', 'meningioma', 'no_tumor', 'pituitary'],
    'early_stopping': 18,   # More patience for extended training
    'min_improvement': 0.001,  # Minimum improvement to reset patience
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Clear memory
torch.cuda.empty_cache() if torch.cuda.is_available() else None
gc.collect()


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class LiteCNN(nn.Module):
    """Extremely lightweight but effective CNN"""
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

# ===== ENHANCED TRAN =====
def get_enhanced_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),  # Added vertical flip
        transforms.RandomRotation(15),       # Increased rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Added affine
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


class MemoryEfficientDataset(Dataset):
    """Dataset that loads images on-the-fly to save memory"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.class_names = CONFIG['class_names']
        self.images = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        print(f"Loading from: {self.data_dir}")
        for idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"WARNING: {class_dir} not found!")
                continue
            
            files = [f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for f in files:
                self.images.append(os.path.join(class_dir, f))
                self.labels.append(idx)
            
            print(f"  {class_name}: {len(files)} images")
        
        print(f"Total: {len(self.images)} images\n")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            # Load image directly without caching
            img = Image.open(self.images[idx]).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                img = self.transform(img)
                
            return img, label
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            # Return a simple fallback
            img = torch.zeros(3, CONFIG['img_size'], CONFIG['img_size'])
            return img, 0


def enhanced_test_time_augmentation(model, test_loader, device):
    """Enhanced TTA with more variations - FIXED VERSION"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
   
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]
    
    with torch.no_grad():
        for batch_data, batch_target in test_loader:
            batch_probs = []
            
            # Process each augmentation
            for transform in tta_transforms:
                transformed_batch = []
                for img in batch_data:
                    # Convert tensor to PIL for transformation - FIXED
                    img_pil = transforms.ToPILImage()(img)
                    transformed_img = transform(img_pil)
                    transformed_batch.append(transformed_img)
                
                transformed_batch = torch.stack(transformed_batch).to(device)
                output = model(transformed_batch)
                probs = F.softmax(output, dim=1)
                batch_probs.append(probs.cpu().numpy())
            
            # Average predictions from all augmentations
            avg_probs = np.mean(batch_probs, axis=0)
            final_preds = np.argmax(avg_probs, axis=1)
            
            all_preds.extend(final_preds)
            all_labels.extend(batch_target.numpy())
            all_probs.extend(avg_probs)
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels, all_probs


class OptimizedTrainer:
    def __init__(self):
        self.device = torch.device(CONFIG['device'])
        self.model = LiteCNN(CONFIG['num_classes']).to(self.device)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'learning_rate': []}
        
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        print(f"Model size: {total_params * 4 / 1024**2:.1f} MB\n")
    
    def create_loaders(self):
        train_transform, val_transform = get_enhanced_transforms()
        
        train_dataset = MemoryEfficientDataset(TRAIN_DATA_PATH, train_transform)
        val_dataset = MemoryEfficientDataset(VAL_DATA_PATH, val_transform)
        test_dataset = MemoryEfficientDataset(TEST_DATA_PATH, val_transform)
        
        # Calculate class weights
        class_counts = [0] * CONFIG['num_classes']
        for label in train_dataset.labels:
            class_counts[label] += 1
        
        total = sum(class_counts)
        class_weights = torch.FloatTensor([total / (CONFIG['num_classes'] * c) for c in class_counts])
        print(f"Class counts: {class_counts}")
        print(f"Class weights: {class_weights.numpy()}\n")
        
        # Use minimal workers to save memory
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        return train_loader, val_loader, test_loader, class_weights
    
    def train(self, train_loader, val_loader, class_weights):
        # Use Focal Loss instead of CrossEntropy
        criterion = FocalLoss(alpha=class_weights.to(self.device), gamma=2.0)
        optimizer = optim.Adam(self.model.parameters(), lr=CONFIG['learning_rate'])
        
        # Enhanced scheduler with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        best_val_acc = 0
        patience_counter = 0
        best_epoch = 0
        
        print(f"{'Epoch':>5} {'Train Loss':>12} {'Train Acc':>11} {'Val Loss':>12} {'Val Acc':>11} {'LR':>10} {'Time':>8}")
        print("-" * 80)
        
        start_time = time.time()
        
        for epoch in range(CONFIG['epochs']):
            epoch_start = time.time()
            
            # Clear memory every few epochs
            if epoch % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Multi-stage learning rate schedule
            if epoch == 25:
                print("\n=== Phase 2: Moderate fine-tuning ===")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = CONFIG['learning_rate'] / 5
            elif epoch == 45:
                print("\n=== Phase 3: Aggressive fine-tuning ===")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = CONFIG['learning_rate'] / 20
            
            # Train
            train_loss, train_acc = self._train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"{epoch+1:5d} {train_loss:12.4f} {train_acc:10.2f}% {val_loss:12.4f} "
                  f"{val_acc:10.2f}% {current_lr:9.2e} {epoch_time:7.1f}s")
            
            # Enhanced early stopping with minimum improvement threshold
            improvement = val_acc - best_val_acc
            if improvement > CONFIG['min_improvement']:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, 'best_enhanced_model.pth')
                patience_counter = 0
                print(f"    ↳ New best! (+{improvement:.3f}%, Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                if improvement > 0:
                    print(f"    ↳ Minor improvement (+{improvement:.3f}%)")
                else:
                    print(f"    ↳ No improvement ({patience_counter}/{CONFIG['early_stopping']})")
            
            # Smart early stopping - consider overall training progress
            if patience_counter >= CONFIG['early_stopping']:
                # Only stop if we're well into training and truly plateaued
                if epoch > 30 and epoch > best_epoch + 10:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    print(f"Best was epoch {best_epoch+1} with {best_val_acc:.2f}%")
                    break
                elif epoch <= 30:
                    print("    ↳ Still in early training, continuing...")
                    patience_counter = CONFIG['early_stopping'] - 5  # Reset partially
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")
        
        # Load best model
        checkpoint = torch.load('best_enhanced_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return best_val_acc
    
    def _train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Clear intermediate variables to save memory
            del output, loss
        
        return total_loss / len(loader), 100 * correct / total
    
    def _validate_epoch(self, loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return total_loss / len(loader), 100 * correct / total
    
    def evaluate_standard(self, test_loader):
        """Standard evaluation without TTA"""
        print("\nEvaluating on test set (Standard)...")
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        
        print(f"\nSTANDARD TEST ACCURACY: {accuracy*100:.2f}%")
        print(classification_report(all_labels, all_preds, target_names=CONFIG['class_names']))
        
        cm = confusion_matrix(all_labels, all_preds)
        return accuracy, all_preds, all_labels, cm
    
    def evaluate_with_tta(self, test_loader):
      """Evaluation with Enhanced Test Time Augmentation - FIXED VERSION"""
      print("\nEvaluating with Enhanced Test Time Augmentation...")
      accuracy, all_preds, all_labels, all_probs = enhanced_test_time_augmentation(
      self.model, test_loader, self.device
      )
    
      print(f"\nENHANCED TTA TEST ACCURACY: {accuracy*100:.2f}%")
      print(classification_report(all_labels, all_preds, target_names=CONFIG['class_names']))
    
      cm = confusion_matrix(all_labels, all_preds)
      return accuracy, all_preds, all_labels, cm, all_probs
    
    def plot_enhanced_results(self, standard_results, tta_results=None):
        """Enhanced plotting with learning rate and comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Training history - Loss
        ax = axes[0, 0]
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val', linewidth=2)
        ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Training history - Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, self.history['train_acc'], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, self.history['val_acc'], 'r-', label='Val', linewidth=2)
        ax.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax = axes[0, 2]
        ax.plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Confusion Matrix (Standard)
        ax = axes[1, 0]
        sns.heatmap(standard_results[3], annot=True, fmt='d', cmap='Blues',
                   xticklabels=CONFIG['class_names'],
                   yticklabels=CONFIG['class_names'], ax=ax)
        ax.set_title('Confusion Matrix (Standard)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Accuracy Comparison
        ax = axes[1, 1]
        methods = ['Standard']
        accuracies = [standard_results[0] * 100]
        
        if tta_results:
            methods.append('Enhanced TTA')
            accuracies.append(tta_results[0] * 100)
        
        bars = ax.bar(methods, accuracies, color=['skyblue', 'lightcoral'])
        ax.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Training progress indicator
        ax = axes[1, 2]
        epochs_trained = len(self.history['train_acc'])
        total_epochs = CONFIG['epochs']
        ax.barh(['Progress'], [epochs_trained], color='lightblue')
        ax.axvline(x=total_epochs, color='red', linestyle='--', label='Total Epochs')
        ax.set_xlim(0, total_epochs + 5)
        ax.set_title(f'Training Progress: {epochs_trained}/{total_epochs} epochs')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('enhanced_results_comparison.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved as 'enhanced_results_comparison.png'")
        plt.show()

# ===== MEMORY MONITOR =====
def check_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    else:
        import psutil
        memory = psutil.virtual_memory()
        print(f"RAM Usage: {memory.percent}%")

# ===== MAIN =====
def main():
    print("="*60)
    print("OPTIMIZED BRAIN TUMOR CLASSIFIER")
    print("Extended Epochs + Enhanced Training")
    print("="*60)
    print(f"Image size: {CONFIG['img_size']}x{CONFIG['img_size']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Max epochs: {CONFIG['epochs']} (with progressive fine-tuning)")
    print(f"Early stopping: {CONFIG['early_stopping']} epochs patience")
    print("="*60 + "\n")
    
    # Check data
    if not os.path.exists(BASE_DATA_PATH):
        print(f"ERROR: Data path not found: {BASE_DATA_PATH}")
        return
    
    # Check memory
    check_memory()
    
    try:
        # Initialize optimized trainer
        trainer = OptimizedTrainer()
        
        # Load data
        train_loader, val_loader, test_loader, class_weights = trainer.create_loaders()
        
        # Train with enhanced strategy
        best_val_acc = trainer.train(train_loader, val_loader, class_weights)
        
        # Evaluate with both methods
        print("\n" + "="*50)
        print("FINAL MODEL EVALUATION")
        print("="*50)
        
        # Standard evaluation
        std_accuracy, std_preds, std_labels, std_cm = trainer.evaluate_standard(test_loader)
        
        # Enhanced TTA evaluation
        tta_accuracy, tta_preds, tta_labels, tta_cm, tta_probs = trainer.evaluate_with_tta(test_loader)
        
        # Plot comprehensive results
        trainer.plot_enhanced_results(
            (std_accuracy, std_preds, std_labels, std_cm),
            (tta_accuracy, tta_preds, tta_labels, tta_cm)
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print(f"Standard Accuracy: {std_accuracy*100:.2f}%")
        print(f"Enhanced TTA Accuracy: {tta_accuracy*100:.2f}%")
        improvement = tta_accuracy - std_accuracy
        print(f"Improvement: {improvement:.3f} ({(improvement/std_accuracy)*100:.2f}%)")
        
        if tta_accuracy >= 0.92:
            print("🎯 TARGET ACHIEVED: 92%+ Accuracy! 🎯")
        elif tta_accuracy >= 0.90:
            print("🎯 TARGET ACHIEVED: 90%+ Accuracy! 🎯")
        else:
            print("Close! Additional recommendations:")
            print("1. Try 80-100 epochs with careful monitoring")
            print("2. Add more data augmentation variations")
            print("3. Consider model ensemble")
        
        print("Model saved as: best_enhanced_model.pth")
        print("="*60)
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n💥 OUT OF MEMORY ERROR!")
            print("Try reducing batch_size to 8 or img_size to 112")
        else:
            raise e

if __name__ == "__main__":
    main()