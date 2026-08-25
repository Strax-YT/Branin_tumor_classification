
import base64
import os

import streamlit as st
import requests
import plotly.graph_objects as go

API_BASE_URL = os.environ.get("NEUROSIGHT_API_URL", "http://127.0.0.1:8000/api/v1")

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
        text-align: center;
        color: white;
    }
    .diagnosis-box.tier-high { background: linear-gradient(135deg, #1e8e5a 0%, #167a4c 100%); }
    .diagnosis-box.tier-medium { background: linear-gradient(135deg, #d99a1b 0%, #b97e0f 100%); }
    .diagnosis-box.tier-low { background: linear-gradient(135deg, #c0392b 0%, #a93226 100%); }
    .stButton button {
        width: 100%;
    }
    [data-testid="column"] {
        gap: 0rem;
    }
    .quality-box {
        border: 1px solid #DCE1E5;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 10px;
        font-size: 0.9rem;
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
    st.caption(f"AI backend: {API_BASE_URL}")
    st.info("For educational and research use only.")


def api_post(endpoint, uploaded_file):
    """POST the uploaded file to the AI backend. Returns (json, error_message)."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")}
    try:
        resp = requests.post(f"{API_BASE_URL}{endpoint}", files=files, timeout=30)
    except requests.exceptions.RequestException:
        return None, (
            f"Couldn't reach the AI backend at {API_BASE_URL}. "
            f"Start it with: `uvicorn api.main:app --host 127.0.0.1 --port 8000`"
        )
    if resp.status_code >= 400:
        detail = resp.json().get("detail", resp.text) if resp.headers.get("content-type", "").startswith("application/json") else resp.text
        return None, detail
    return resp.json(), None


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

# File Upload Section - Compact
st.subheader("📤 Upload MRI Scan")
uploaded_file = st.file_uploader("Choose brain MRI image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if uploaded_file:
    quality, error = api_post("/validate-image", uploaded_file)
    if error:
        st.error(f"❌ {error}")
        st.stop()

    st.markdown(
        f"""
        <div class="quality-box">
            <b>MRI Quality Assessment</b> — {quality['status']} ({quality['overall_score']:.0f}/100)<br>
            Resolution: {'Good' if quality['resolution_ok'] else 'Low'} ({quality['width']}×{quality['height']})
            &nbsp;·&nbsp; Sharpness: {quality['sharpness_score']:.0f}/100
            &nbsp;·&nbsp; Contrast: {quality['contrast_score']:.0f}/100
            &nbsp;·&nbsp; Brightness: {quality['brightness']:.0f}/255
        </div>
        """,
        unsafe_allow_html=True,
    )
    if quality["status"] == "Poor":
        st.warning(
            "⚠️ This image scores low on resolution/sharpness/contrast/brightness. "
            "Analysis will still run, but treat the result with extra caution."
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, use_container_width=True)

    with col2:
        if st.button("🚀 Start AI Analysis", type="primary", use_container_width=True):
            with st.spinner("AI analyzing MRI scan..."):
                analysis, error = api_post("/analyze", uploaded_file)
                if error:
                    st.error(f"Analysis error: {error}")
                else:
                    st.session_state.analysis = analysis
                    st.session_state.analysis_complete = True

    # Results Display - Compact
    if st.session_state.analysis_complete:
        analysis = st.session_state.analysis
        class_names = list(analysis['probabilities'].keys())
        probabilities = list(analysis['probabilities'].values())
        predicted_class = class_names.index(analysis['prediction'])
        confidence = analysis['confidence']
        uncertainty = analysis['uncertainty']

        # Diagnosis Box - tiered by the Safety Decision Engine
        tier = uncertainty['tier']
        tier_icon = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[tier]
        tier_css = {"High": "tier-high", "Medium": "tier-medium", "Low": "tier-low"}[tier]

        extra_line = ""
        if tier != "High":
            extra_line = (
                f"<p style='margin:4px 0 0'>Alternative possibility: "
                f"{uncertainty['runner_up_class']} ({uncertainty['runner_up_probability']:.1%})</p>"
            )

        st.markdown(f"""
        <div class="diagnosis-box {tier_css}">
            <h2>AI Diagnosis: {analysis['prediction']}</h2>
            <h3>{tier_icon} Confidence: {confidence:.2%} &nbsp;·&nbsp; Reliability: {uncertainty['reliability_score']:.0f}/100 ({tier})</h3>
            <p style="margin:4px 0 0">{uncertainty['message']}</p>
            {extra_line}
        </div>
        """, unsafe_allow_html=True)

        # Results in Tabs - No Empty Space
        tab1, tab2, tab3 = st.tabs(["📊 Probabilities", "🔍 Confidence", "🎯 Attention"])

        with tab1:
            fig_prob = create_probability_chart(probabilities, class_names, predicted_class)
            st.plotly_chart(fig_prob, use_container_width=True)

        with tab2:
            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                fig_gauge = create_confidence_gauge(confidence)
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col_conf2:
                st.metric("Prediction Entropy", f"{uncertainty['entropy']:.3f}")
                st.metric("Top-2 Gap", f"{uncertainty['top_two_gap']:.3f}")
                st.metric("Reliability Tier", tier)

        with tab3:
            overlay_b64 = analysis.get('attention_overlay_png_b64')
            if overlay_b64:
                st.image(base64.b64decode(overlay_b64), use_container_width=True)
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

# Minimal Footer
st.markdown("---")
st.caption("Brain Tumor AI Classification System • 91.62% Accuracy • Educational Use Only")
