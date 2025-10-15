"""
Simple Streamlit App - Focus on Working First
"""

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import os

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Classification")
st.markdown("**Simple & Reliable Version**")

# Import from the universal module
try:
    from universal_explainable_model import load_original_model, UniversalExplainabilityEngine
    IMPORT_SUCCESS = True
except Exception as e:
    st.error(f"❌ Import error: {e}")
    IMPORT_SUCCESS = False

@st.cache_resource
def load_model_simple():
    """Simple model loading"""
    if not IMPORT_SUCCESS:
        return None
        
    try:
        model = load_original_model()
        if model:
            st.sidebar.success("✅ Model loaded!")
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

def preprocess_image_simple(image):
    """Simple image preprocessing"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image).unsqueeze(0)

# Sidebar
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Upload a brain MRI image
2. Click 'Analyze' 
3. View results and confidence
""")

# Main app
if IMPORT_SUCCESS:
    model = load_model_simple()
    
    if model:
        uploaded_file = st.file_uploader("Choose MRI image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
                
                if st.button("Analyze Image", type="primary"):
                    with st.spinner("Processing..."):
                        try:
                            # Preprocess
                            input_tensor = preprocess_image_simple(image)
                            
                            # Get prediction
                            with torch.no_grad():
                                output = model(input_tensor)
                                probabilities = F.softmax(output, dim=1).numpy()[0]
                                predicted_class = np.argmax(probabilities)
                                confidence = probabilities[predicted_class]
                            
                            # Store results
                            st.session_state.probabilities = probabilities
                            st.session_state.predicted_class = predicted_class
                            st.session_state.confidence = confidence
                            
                            # Try explainability
                            try:
                                explainer = UniversalExplainabilityEngine(
                                    model, 
                                    ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
                                )
                                cam, _ = explainer.generate_grad_cam(input_tensor)
                                overlay = explainer.create_attention_overlay(image, cam)
                                st.session_state.overlay = overlay
                                st.session_state.explainability_works = True
                            except:
                                st.session_state.explainability_works = False
                            
                        except Exception as e:
                            st.error(f"Analysis error: {e}")
            
            with col2:
                if 'probabilities' in st.session_state:
                    st.subheader("Results")
                    
                    class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
                    predicted_name = class_names[st.session_state.predicted_class]
                    
                    st.success(f"**Diagnosis:** {predicted_name}")
                    st.info(f"**Confidence:** {st.session_state.confidence:.2%}")
                    
                    # Show probabilities
                    st.subheader("Probabilities")
                    for i, (name, prob) in enumerate(zip(class_names, st.session_state.probabilities)):
                        st.write(f"{name}: {prob:.2%}")
                        st.progress(float(prob))
                    
                    # Show explainability if available
                    if (hasattr(st.session_state, 'explainability_works') and 
                        st.session_state.explainability_works and 
                        'overlay' in st.session_state):
                        
                        st.subheader("AI Attention Map")
                        st.image(st.session_state.overlay, 
                                caption="Where the AI focuses",
                                use_column_width=True)
                    else:
                        st.info("Explainability features not available")
    
    else:
        st.error("Could not load model. Check console for details.")
else:
    st.error("Module import failed. Check the universal_explainable_model.py file.")

st.markdown("---")
st.markdown("Brain Tumor Classification System")