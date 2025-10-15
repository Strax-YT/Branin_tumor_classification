"""
Explainable Brain Tumor Classification Web App
With Grad-CAM, Feature Analysis, and Confidence Metrics
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import cv2

# Import explainability modules
from explainable_model import ExplainableLiteCNN, ExplainabilityEngine, create_saliency_map

# ===== MODEL LOADING =====
@st.cache_resource
def load_explainable_model():
    """Load the explainable model"""
    try:
        model = ExplainableLiteCNN(num_classes=4)
        
        model_paths = ["best_lite_model.pth", "../models/best_lite_model.pth"]
        model_loaded = False
        
        for path in model_paths:
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                model_loaded = True
                st.sidebar.success(f"✅ Explainable model loaded from: {path}")
                break
        
        if not model_loaded:
            st.error("❌ Model file not found.")
            return None, None
        
        model.eval()
        explainer = ExplainabilityEngine(model, 
            ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'])
        
        return model, explainer
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

def main():
    st.set_page_config(
        page_title="Explainable Brain Tumor Classifier",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .explanation-section {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧠 Explainable Brain Tumor Classification</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔍 Explainability Features")
    st.sidebar.markdown("""
    **Visual Explanations:**
    - 🎯 **Grad-CAM**: See where the model focuses
    - 🔍 **Saliency Maps**: Important pixels for decision
    - 📊 **Feature Importance**: Which features matter most
    - ⚖️ **Confidence Analysis**: Model certainty metrics
    - ❓ **Uncertainty Measurement**: When the model is unsure
    """)
    
    st.sidebar.title("📈 Model Metrics")
    st.sidebar.write("""
    - **Overall Accuracy**: 91.62%
    - **Glioma Precision**: 95%
    - **Meningioma Precision**: 80%
    - **No Tumor Precision**: 99%
    - **Pituitary Precision**: 94%
    """)
    
    # Load model
    model, explainer = load_explainable_model()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload MRI Scan")
        uploaded_file = st.file_uploader(
            "Choose a brain MRI image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear MRI scan for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original MRI Scan", use_column_width=True)
            
            # Explanation level selector
            st.subheader("🔧 Explanation Settings")
            explanation_depth = st.select_slider(
                "Explanation Detail Level",
                options=['Basic', 'Detailed', 'Comprehensive'],
                value='Detailed'
            )
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if uploaded_file is not None and model is not None and explainer is not None:
            with st.spinner("🔄 Analyzing with explainable AI..."):
                # Preprocess and predict
                image_tensor = preprocess_image(image)
                
                # Get prediction and explanations
                cam, output = explainer.generate_grad_cam(image_tensor)
                probabilities = F.softmax(output, dim=1).numpy()[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
                
                # Generate saliency map
                saliency = create_saliency_map(model, image_tensor, predicted_class)
                
                time.sleep(1)
            
            # Display main prediction
            class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
            class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"
            
            st.markdown(f"""
            <div class="explanation-section">
                <h3>🎯 Primary Diagnosis: {class_names[predicted_class]}</h3>
                <h4 class="{confidence_class}">Confidence: {confidence:.2%}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Basic explanations (always shown)
            st.subheader("📊 Basic Explanations")
            
            # Probability distribution
            fig_basic = go.Figure(data=[
                go.Bar(x=class_names, y=probabilities,
                      marker_color=[class_colors[predicted_class] if i == predicted_class else 'lightgray' 
                                  for i in range(4)])
            ])
            fig_basic.update_layout(title="Prediction Probabilities", height=300)
            st.plotly_chart(fig_basic, use_container_width=True)
            
            if explanation_depth in ['Detailed', 'Comprehensive']:
                st.subheader("🎯 Detailed Explanations")
                
                # Grad-CAM visualization
                col_cam1, col_cam2 = st.columns(2)
                
                with col_cam1:
                    # Create overlay
                    overlay = explainer.create_attention_overlay(
                        image.resize((128, 128)), cam
                    )
                    st.image(overlay, caption="Grad-CAM Attention Map", use_column_width=True)
                
                with col_cam2:
                    st.image(saliency, caption="Saliency Map (Important Pixels)", 
                            use_column_width=True, clamp=True)
                
                st.markdown("""
                **Grad-CAM Explanation:** 
                - 🔴 **Red areas**: High attention - model focuses here
                - 🔵 **Blue areas**: Low attention - less important for decision
                """)
            
            if explanation_depth == 'Comprehensive':
                st.subheader("📈 Comprehensive Analysis")
                
                # Generate full explanation report
                explanation_fig, confidence_info = explainer.generate_explanation_report(
                    image_tensor, predicted_class, confidence, probabilities, cam
                )
                
                st.plotly_chart(explanation_fig, use_container_width=True)
                
                # Confidence metrics
                col_met1, col_met2, col_met3 = st.columns(3)
                
                with col_met1:
                    st.metric("Prediction Entropy", f"{confidence_info['entropy']:.3f}",
                             help="Lower entropy = more confident decision")
                
                with col_met2:
                    st.metric("Confidence Gap", f"{confidence_info['confidence_gap']:.3f}",
                             help="Difference between top 2 predictions")
                
                with col_met3:
                    uncertainty_status = "High" if confidence_info['uncertainty'] else "Low"
                    st.metric("Uncertainty", uncertainty_status,
                             help="Model's self-assessment of certainty")
                
                # Feature insights
                st.subheader("🔬 Model Decision Insights")
                
                if confidence_info['uncertainty']:
                    st.warning("""
                    **High Uncertainty Detected:** The model is less confident about this prediction.
                    This could be due to:
                    - Unusual image characteristics
                    - Multiple possible diagnoses
                    - Image quality issues
                    """)
                
                # Decision factors
                st.info("""
                **Key Decision Factors:**
                - The model focuses on tumor location and texture patterns
                - Shape regularity and boundary clarity affect confidence
                - Contrast enhancement patterns are important indicators
                """)
            
            # Medical disclaimer
            st.warning("""
            **Medical Disclaimer:** This AI tool provides explanations for educational purposes. 
            The attention maps show where the model focuses, but may not align with clinical expertise.
            Always consult healthcare professionals for medical diagnoses.
            """)
            
        elif uploaded_file is not None:
            st.error("❌ Model not loaded properly.")
        else:
            st.info("👆 Upload an MRI scan to see AI explanations")
            
            # Demo of explainability features
            st.subheader("🎯 What Explainability Adds")
            col_demo1, col_demo2, col_demo3 = st.columns(3)
            
            with col_demo1:
                st.markdown("""
                **🎯 Attention Maps**
                - See where AI focuses
                - Understand decision regions
                - Visual model reasoning
                """)
            
            with col_demo2:
                st.markdown("""
                **📊 Confidence Metrics**
                - Prediction certainty
                - Uncertainty measurement
                - Reliability indicators
                """)
            
            with col_demo3:
                st.markdown("""
                **🔍 Feature Analysis**
                - Important patterns
                - Decision factors
                - Model insights
                """)

if __name__ == "__main__":
    main()