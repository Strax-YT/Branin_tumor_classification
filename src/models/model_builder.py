import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pandas as pd
from PIL import Image
import os
import glob
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

class BrainTumorClassifier:
    def __init__(self, img_height=64, img_width=64):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = 4
        self.class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
        self.models = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=100)  # Reduce dimensionality
        self.training_history = {}
        
    def load_images_from_folder(self, folder_path, max_images_per_class=None):
        """Load images from your actual folder structure"""
        images = []
        labels = []
        loaded_counts = {}
        
        print(f"Loading images from: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"ERROR: Folder {folder_path} does not exist!")
            print("Please check your folder path.")
            return np.array([]), np.array([])
        
        for class_idx, class_name in enumerate(self.class_names):
            class_folder = os.path.join(folder_path, class_name)
            
            if not os.path.exists(class_folder):
                print(f"WARNING: Class folder {class_folder} not found!")
                loaded_counts[class_name] = 0
                continue
            
            # Look for image files with common extensions
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(class_folder, ext)))
            
            if max_images_per_class:
                image_files = image_files[:max_images_per_class]
            
            class_image_count = 0
            
            for img_path in image_files:
                try:
                    # Load and preprocess image
                    img = Image.open(img_path)
                    
                    # Convert to RGB if not already
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize image
                    img = img.resize((self.img_width, self.img_height))
                    
                    # Convert to array and normalize
                    img_array = np.array(img).flatten() / 255.0
                    
                    images.append(img_array)
                    labels.append(class_idx)
                    class_image_count += 1
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            loaded_counts[class_name] = class_image_count
            print(f"Loaded {class_image_count} images for {class_name}")
        
        print(f"Total images loaded: {len(images)}")
        print("Class distribution:", loaded_counts)
        
        if len(images) == 0:
            print("ERROR: No images were loaded! Please check:")
            print("1. Folder paths are correct")
            print("2. Image files exist in the folders")
            print("3. Image file extensions are supported")
            return np.array([]), np.array([])
        
        return np.array(images), np.array(labels)
    
    def prepare_data(self, train_dir, val_dir, test_dir):
        """Prepare training, validation, and test data from your folders"""
        
        print("="*60)
        print("LOADING YOUR BRAIN TUMOR DATASET")
        print("="*60)
        
        print("\n1. Loading TRAINING data...")
        X_train, y_train = self.load_images_from_folder(train_dir)
        
        print("\n2. Loading VALIDATION data...")
        X_val, y_val = self.load_images_from_folder(val_dir)
        
        print("\n3. Loading TEST data...")
        X_test, y_test = self.load_images_from_folder(test_dir)
        
        # Check if data was loaded successfully
        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            print("ERROR: Could not load data from the specified directories!")
            return None, None, None
        
        print("\n4. Preprocessing data...")
        print("   - Scaling features...")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("   - Applying PCA for dimensionality reduction...")
        
        # Apply PCA for dimensionality reduction
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_val_pca = self.pca.transform(X_val_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        print(f"   - Reduced features from {X_train_scaled.shape[1]} to {X_train_pca.shape[1]}")
        
        return (X_train_pca, y_train), (X_val_pca, y_val), (X_test_pca, y_test)
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and track performance"""
        
        print("\n" + "="*60)
        print("TRAINING MODELS")
        print("="*60)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Calculate metrics
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            # Cross-validation
            print(f"   Running cross-validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"   Results -> Train: {train_acc:.4f}, Val: {val_acc:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        
        # Select best model based on validation accuracy
        best_model_name = max(results, key=lambda x: results[x]['val_accuracy'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 BEST MODEL: {best_model_name} (Val Accuracy: {results[best_model_name]['val_accuracy']:.4f})")
        
        # Store results for plotting
        self.training_results = results
        
        return results
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the best model on test data"""
        
        if not hasattr(self, 'best_model'):
            print("No trained model found. Train models first.")
            return None
        
        print(f"\n" + "="*60)
        print(f"EVALUATING BEST MODEL: {self.best_model_name}")
        print("="*60)
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, average='weighted'),
            'test_recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        print(f"Test Accuracy:  {metrics['test_accuracy']:.4f}")
        print(f"Test Precision: {metrics['test_precision']:.4f}")
        print(f"Test Recall:    {metrics['test_recall']:.4f}")
        print(f"F1-Score:       {metrics['f1_score']:.4f}")
        
        return metrics, y_pred, y_pred_proba
    
    def plot_model_comparison(self):
        """Plot comparison of different models"""
        
        if not hasattr(self, 'training_results'):
            print("No training results available.")
            return
        
        models = list(self.training_results.keys())
        train_accs = [self.training_results[m]['train_accuracy'] for m in models]
        val_accs = [self.training_results[m]['val_accuracy'] for m in models]
        cv_means = [self.training_results[m]['cv_mean'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, train_accs, width, label='Training Accuracy', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x, val_accs, width, label='Validation Accuracy', alpha=0.8, color='#A23B72')
        bars3 = ax.bar(x + width, cv_means, width, label='CV Accuracy', alpha=0.8, color='#F18F01')
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return cm
    
    def plot_classification_report(self, y_true, y_pred):
        """Plot classification report as heatmap"""
        
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        # Convert to DataFrame for visualization
        df_report = pd.DataFrame(report).T
        df_class_report = df_report.iloc[:-3, :-1]  # Remove support column and summary rows
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_class_report, annot=True, cmap='Blues', fmt='.3f',
                   cbar_kws={'label': 'Score'})
        plt.title('Classification Report Heatmap', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Classes', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        return report
    
    def plot_metrics_summary(self, metrics):
        """Plot all metrics in a summary chart"""
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics['test_accuracy'], metrics['test_precision'], 
                        metrics['test_recall'], metrics['f1_score']]
        
        plt.figure(figsize=(12, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        plt.title(f'Model Performance Metrics - {self.best_model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
    
    def plot_dataset_distribution(self, train_counts, val_counts, test_counts):
        """Plot dataset distribution"""
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        train_values = [train_counts.get(name, 0) for name in self.class_names]
        val_values = [val_counts.get(name, 0) for name in self.class_names]
        test_values = [test_counts.get(name, 0) for name in self.class_names]
        
        ax.bar(x - width, train_values, width, label='Training', color=colors[0], alpha=0.8)
        ax.bar(x, val_values, width, label='Validation', color=colors[1], alpha=0.8)
        ax.bar(x + width, test_values, width, label='Testing', color=colors[2], alpha=0.8)
        
        ax.set_xlabel('Brain Tumor Classes', fontsize=12)
        ax.set_ylabel('Number of Images', fontsize=12)
        ax.set_title('Your Dataset Distribution', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (train_val, val_val, test_val) in enumerate(zip(train_values, val_values, test_values)):
            if train_val > 0:
                ax.text(i - width, train_val + max(train_values) * 0.01, str(train_val), 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            if val_val > 0:
                ax.text(i, val_val + max(val_values) * 0.01, str(val_val), 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            if test_val > 0:
                ax.text(i + width, test_val + max(test_values) * 0.01, str(test_val), 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

def run_brain_tumor_classification(train_dir, val_dir, test_dir):
    """Main function to run the complete classification pipeline"""
    
    print("🧠 BRAIN TUMOR CLASSIFICATION PIPELINE")
    print("="*60)
    
    # Initialize classifier
    classifier = BrainTumorClassifier(img_height=64, img_width=64)
    
    # Load and prepare data
    data = classifier.prepare_data(train_dir, val_dir, test_dir)
    
    if data[0] is None:
        print("❌ Failed to load data. Please check your folder paths.")
        return None
    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = data
    
    # Train models
    results = classifier.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on test data
    metrics, y_pred, y_pred_proba = classifier.evaluate_model(X_test, y_test)
    
    # Generate all plots
    print(f"\n" + "="*60)
    print("GENERATING VISUALIZATION CHARTS")
    print("="*60)
    
    print("1. Model Comparison Chart...")
    classifier.plot_model_comparison()
    
    print("2. Confusion Matrix...")
    classifier.plot_confusion_matrix(y_test, y_pred)
    
    print("3. Classification Report...")
    classifier.plot_classification_report(y_test, y_pred)
    
    print("4. Performance Metrics Summary...")
    classifier.plot_metrics_summary(metrics)
    
    return classifier, metrics, y_pred

# EXAMPLE USAGE WITH YOUR DATA
if __name__ == "__main__":
    
    print("🚀 TO USE WITH YOUR ACTUAL DATA:")
    print("="*60)
    

train_dir = 'D:\Branin_tumor_classification\data\train'
val_dir = 'D:\Branin_tumor_classification\data\val'        # Should contain: glioma/, meningioma/, no_tumor/, pituitary/
test_dir = 'D:\Branin_tumor_classification\data\test'      # Should contain: glioma/, meningioma/, no_tumor/, pituitary/

# Run the complete pipeline:
classifier, metrics, predictions = run_brain_tumor_classification(train_dir, val_dir, test_dir)

# Your results will include:
# - Model comparison charts
# - Confusion matrix 
# - Classification report
# - Performance metrics
# - Accuracy visualization
    """)
    
    print("\n📁 EXPECTED FOLDER STRUCTURE:")
    
    print("\n💡 QUICK START:")
    print("1. Update the folder paths above to your actual data location")
    print("2. Run: classifier, metrics, predictions = run_brain_tumor_classification(train_dir, val_dir, test_dir)")
    print("3. All charts and accuracy metrics will be displayed automatically!")