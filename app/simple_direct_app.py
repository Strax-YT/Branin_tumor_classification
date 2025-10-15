"""
COMPACT Brain Tumor Classification App
No Empty Spaces - Clean Layout
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import plotly.graph_objects as go
import plotly.express as px
import cv2
import time

from direct_model_loader import load_direct_model, ExplainabilityWrapper

# Page config
st.set_page_config(
    page_title="Brain Tumor AI Classifier",
    page_icon="🧠",
    layout="wide"
)

# Compact CSS - Remove all extra spaces
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    .stApp {
        margin-top: -50px;
    }
    .diagnosis-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
    }
    .stButton button {
        width: 100%;
    }
    [data-testid="column"] {
        gap: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# Header - Compact
st.markdown('<h1 class="main-header">🧠 Brain Tumor AI Classifier</h1>', unsafe_allow_html=True)
st.markdown("**91.62% Accuracy • Explainable AI**")

# Sidebar - Ultra Compact
with st.sidebar:
    st.title("🔬 System Info")
    st.metric("Overall Accuracy", "91.62%")
    st.write("**Class Performance:**")
    st.write("• Glioma: 95% precision")
    st.write("• Meningioma: 80% precision")  
    st.write("• No Tumor: 99% precision")
    st.write("• Pituitary: 94% precision")
    st.markdown("---")
    st.info("For educational and research use only.")

# Load model
@st.cache_resource
def load_model():
    return load_direct_model()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def create_probability_chart(probabilities, class_names, predicted_class):
    colors = ['#FF6B6B' if i == predicted_class else '#4ECDC4' for i in range(len(class_names))]
    fig = go.Figure(data=[
        go.Bar(x=class_names, y=probabilities, marker_color=colors,
               text=[f'{p:.2%}' for p in probabilities], textposition='auto')
    ])
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Tumor Types",
        yaxis_title="Probability", 
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        height=300
    )
    return fig

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}]
        }
    ))
    fig.update_layout(height=250)
    return fig

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Main App - No Empty Space
model = load_model()

if model:
    # File Upload Section - Compact
    st.subheader("📤 Upload MRI Scan")
    uploaded_file = st.file_uploader("Choose brain MRI image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
        with col2:
            if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
                with st.spinner("AI analyzing MRI scan..."):
                    try:
                        input_tensor = preprocess_image(image)
                        class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
                        explainer = ExplainabilityWrapper(model, class_names)
                        result = explainer.predict_with_explanations(input_tensor)
                        confidence_info = explainer.analyze_confidence(result['probabilities'])
                        
                        overlay = None
                        if result['attention_map'] is not None:
                            overlay = explainer.create_attention_overlay(image, result['attention_map'])
                        
                        st.session_state.result = result
                        st.session_state.confidence_info = confidence_info
                        st.session_state.overlay = overlay
                        st.session_state.class_names = class_names
                        st.session_state.analysis_complete = True
                        
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
        
        # Results Display - Compact
        if st.session_state.analysis_complete:
            result = st.session_state.result
            class_names = st.session_state.class_names
            confidence_info = st.session_state.confidence_info
            
            # Diagnosis Box
            predicted_name = class_names[result['predicted_class']]
            confidence = result['confidence']
            
            confidence_color = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
            
            st.markdown(f"""
            <div class="diagnosis-box">
                <h2>AI Diagnosis: {predicted_name}</h2>
                <h3>{confidence_color} Confidence: {confidence:.2%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Results in Tabs - No Empty Space
            tab1, tab2, tab3 = st.tabs(["📊 Probabilities", "🔍 Confidence", "🎯 Attention"])
            
            with tab1:
                fig_prob = create_probability_chart(result['probabilities'], class_names, result['predicted_class'])
                st.plotly_chart(fig_prob, use_container_width=True)
                
            with tab2:
                col_conf1, col_conf2 = st.columns(2)
                with col_conf1:
                    fig_gauge = create_confidence_gauge(confidence)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                with col_conf2:
                    st.metric("Prediction Entropy", f"{confidence_info['entropy']:.3f}")
                    st.metric("Confidence Gap", f"{confidence_info['confidence_gap']:.3f}")
                    st.metric("Uncertainty", "High" if confidence_info['uncertainty'] else "Low")
            
            with tab3:
                if st.session_state.overlay is not None:
                    st.image(st.session_state.overlay, use_column_width=True)
                    st.caption("🔴 Red: High attention | 🟡 Yellow: Medium | 🔵 Blue: Low")
                else:
                    st.info("Attention visualization not available")
            
            # Medical Notes - Compact
            with st.expander("💡 Medical Guidance"):
                st.write("""
                **Clinical Recommendations:**
                • Consult with neurologist for proper diagnosis
                • Consider additional imaging if recommended
                • This AI analysis is for educational purposes
                • Always seek professional medical advice
                """)
    
    else:
        # Welcome Section - Compact
        st.info("👆 Upload a brain MRI image above to begin analysis")
        
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            st.write("**🎯 AI Features:**")
            st.write("• Grad-CAM Attention Maps")
            st.write("• Confidence Analysis")
            st.write("• Probability Distribution")
            st.write("• Uncertainty Measurement")
            
        with col_feat2:
            st.write("**🚀 Quick Process:**")
            st.write("1. Upload MRI Image")
            st.write("2. Click Analyze Button") 
            st.write("3. View AI Results")
            st.write("4. Understand Diagnosis")

else:
    st.error("❌ Model failed to load. Check model file.")

# Minimal Footer
st.markdown("---")
st.caption("Brain Tumor AI Classification System • 91.62% Accuracy • Educational Use Only")