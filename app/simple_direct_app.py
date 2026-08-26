
import base64
import os

import streamlit as st
import requests
import plotly.graph_objects as go

API_BASE_URL = os.environ.get("NEUROSIGHT_API_URL", "http://127.0.0.1:8000/api/v1")

INK = "#16202A"
MUTED = "#5C6B78"
LINE = "#DCE3E8"
SURFACE = "#FFFFFF"
ACCENT = "#2C5F82"
ACCENT_LIGHT = "#E8F1F8"
GOOD = "#2F7D5A"
WARN = "#A9720B"
CRITICAL = "#B33A3A"

TIER_COLOR = {"High": GOOD, "Medium": WARN, "Low": CRITICAL}

# Page config
st.set_page_config(
    page_title="NeuroSight",
    page_icon="\U0001FA7A",
    layout="wide",
)

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
        color: {INK};
    }}
    .block-container {{
        padding-top: 2rem;
        max-width: 1100px;
    }}
    .mono {{
        font-family: 'IBM Plex Mono', monospace;
        font-variant-numeric: tabular-nums;
    }}

    .header-banner {{
        background: {ACCENT_LIGHT};
        border: 1px solid {LINE};
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.6rem;
    }}
    .app-header {{
        display: flex;
        align-items: baseline;
        gap: 0.6rem;
        margin-bottom: 0.15rem;
    }}
    .app-header .name {{
        font-size: 1.9rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: {ACCENT};
    }}
    .app-header .tagline {{
        color: {MUTED};
        font-size: 0.95rem;
    }}
    .app-sub {{
        color: {MUTED};
        font-size: 0.88rem;
    }}

    .card {{
        background: {SURFACE};
        border: 1px solid {LINE};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.9rem;
    }}
    .card .card-label {{
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {MUTED};
        font-weight: 600;
        margin-bottom: 0.4rem;
    }}

    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
    }}
    .badge .dot {{
        width: 7px; height: 7px; border-radius: 50%;
    }}

    .diagnosis-card {{
        background: {SURFACE};
        border: 1px solid {LINE};
        border-left: 5px solid var(--tier-color, {ACCENT});
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin: 0.6rem 0 1rem;
    }}
    .diagnosis-card .eyebrow {{
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: {MUTED};
        font-weight: 600;
    }}
    .diagnosis-card h2 {{
        margin: 0.15rem 0 0.3rem;
        font-size: 1.5rem;
        font-weight: 600;
    }}
    .diagnosis-card .meta {{
        color: {MUTED};
        font-size: 0.9rem;
    }}
    .diagnosis-card .message {{
        margin-top: 0.6rem;
        font-size: 0.92rem;
    }}
    .diagnosis-card .alt {{
        margin-top: 0.3rem;
        font-size: 0.86rem;
        color: {MUTED};
    }}

    .legend-item {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.82rem;
        color: {MUTED};
        margin-right: 1rem;
    }}
    .legend-swatch {{
        width: 10px; height: 10px; border-radius: 2px; display: inline-block;
    }}

    .stButton button[kind="primary"] {{
        background-color: {ACCENT};
        border-color: {ACCENT};
    }}
    .stButton button {{
        width: 100%;
    }}
    [data-testid="column"] {{
        gap: 0.75rem;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="header-banner">
        <div class="app-header">
            <span class="name">NeuroSight</span>
            <span class="tagline">Explainable brain MRI classification</span>
        </div>
        <div class="app-sub">91.62% verified test accuracy &nbsp;&middot;&nbsp; \
LiteCNN &nbsp;&middot;&nbsp; Grad-CAM explainability</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("**System Info**")
    st.metric("Overall Accuracy", "91.62%")
    st.caption("Per-class precision")
    st.write("Glioma — 95%")
    st.write("Meningioma — 80%")
    st.write("No Tumor — 99%")
    st.write("Pituitary — 94%")
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


def api_get(endpoint, params=None):
    """GET from the AI backend. Returns (json, error_message)."""
    try:
        resp = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=15)
    except requests.exceptions.RequestException:
        return None, f"Couldn't reach the AI backend at {API_BASE_URL}."
    if resp.status_code >= 400:
        return None, resp.text
    return resp.json(), None


def create_probability_chart(probabilities, class_names, predicted_class):
    colors = [ACCENT if i == predicted_class else "#B7C6D1" for i in range(len(class_names))]
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
        height=300,
        font=dict(family="IBM Plex Sans, sans-serif", color=INK),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': ACCENT},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 60], 'color': "#F0E6D6"},
                {'range': [60, 80], 'color': "#E7EEDD"},
                {'range': [80, 100], 'color': "#DCEAE1"}]
        }
    ))
    fig.update_layout(
        height=250,
        font=dict(family="IBM Plex Sans, sans-serif", color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Prediction History & Metrics - Collapsed by default
with st.expander("Prediction History & Metrics", expanded=False):
    metrics, metrics_error = api_get("/metrics")
    if metrics_error:
        st.caption(f"Backend unreachable — {metrics_error}")
    elif metrics["total_predictions"] == 0:
        st.caption("No predictions recorded yet.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Predictions", metrics["total_predictions"])
        m2.metric("Avg Inference Time", f"{metrics['average_inference_time_ms']:.0f} ms")
        m3.metric("Avg Confidence", f"{metrics['average_confidence']:.1%}")
        review_rate = metrics["review_recommended_count"] / metrics["total_predictions"]
        m4.metric("Review-Recommended Rate", f"{review_rate:.1%}")

        history, history_error = api_get("/history", params={"limit": 20})
        if history_error:
            st.caption(f"Could not load history — {history_error}")
        elif history["items"]:
            st.dataframe(
                [
                    {
                        "Timestamp": item["timestamp"],
                        "Prediction": item["predicted_class"],
                        "Confidence": f"{item['confidence']:.1%}",
                        "Reliability": item["reliability_tier"],
                        "Review advised": "Yes" if item["review_recommended"] else "No",
                        "Time (ms)": round(item["inference_time_ms"], 1),
                    }
                    for item in history["items"]
                ],
                use_container_width=True,
                hide_index=True,
            )

# File Upload Section
st.subheader("Upload MRI Scan")
uploaded_file = st.file_uploader("Choose brain MRI image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

if uploaded_file:
    quality, error = api_post("/validate-image", uploaded_file)
    if error:
        st.error(f"Couldn't read that file — {error}")
        st.stop()

    quality_color = {"Suitable": GOOD, "Marginal": WARN, "Poor": CRITICAL}[quality["status"]]
    st.markdown(
        f"""
        <div class="card">
            <div class="card-label">MRI Quality Assessment</div>
            <span class="badge" style="background:{quality_color}18; color:{quality_color};">
                <span class="dot" style="background:{quality_color};"></span>{quality['status']}
            </span>
            <span class="mono" style="margin-left:0.5rem; color:{MUTED};">{quality['overall_score']:.0f}/100</span>
            <div style="margin-top:0.5rem; color:{MUTED}; font-size:0.86rem;">
                Resolution: {'Good' if quality['resolution_ok'] else 'Low'} ({quality['width']}&times;{quality['height']})
                &nbsp;&middot;&nbsp; Sharpness: <span class="mono">{quality['sharpness_score']:.0f}</span>/100
                &nbsp;&middot;&nbsp; Contrast: <span class="mono">{quality['contrast_score']:.0f}</span>/100
                &nbsp;&middot;&nbsp; Brightness: <span class="mono">{quality['brightness']:.0f}</span>/255
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if quality["status"] == "Poor":
        st.warning(
            "This image scores low on resolution/sharpness/contrast/brightness. "
            "Analysis will still run, but treat the result with extra caution."
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, use_column_width=True)

    with col2:
        if st.button("Start AI Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing MRI scan..."):
                analysis, error = api_post("/analyze", uploaded_file)
                if error:
                    st.error(f"Analysis error: {error}")
                else:
                    st.session_state.analysis = analysis
                    st.session_state.analysis_complete = True

    # Results Display
    if st.session_state.analysis_complete:
        analysis = st.session_state.analysis
        class_names = list(analysis['probabilities'].keys())
        probabilities = list(analysis['probabilities'].values())
        predicted_class = class_names.index(analysis['prediction'])
        confidence = analysis['confidence']
        uncertainty = analysis['uncertainty']

        tier = uncertainty['tier']
        tier_color = TIER_COLOR[tier]

        extra_line = ""
        if tier != "High":
            extra_line = (
                f"<div class='alt'>Alternative possibility: "
                f"{uncertainty['runner_up_class']} ({uncertainty['runner_up_probability']:.1%})</div>"
            )

        st.markdown(f"""
        <div class="diagnosis-card" style="--tier-color:{tier_color};">
            <div class="eyebrow">AI Diagnosis</div>
            <h2>{analysis['prediction']}</h2>
            <div class="meta">
                Confidence <span class="mono">{confidence:.1%}</span>
                &nbsp;&middot;&nbsp;
                <span class="badge" style="background:{tier_color}18; color:{tier_color};">
                    <span class="dot" style="background:{tier_color};"></span>{tier} reliability
                </span>
                &nbsp;&middot;&nbsp;
                Score <span class="mono">{uncertainty['reliability_score']:.0f}/100</span>
            </div>
            <div class="message">{uncertainty['message']}</div>
            {extra_line}
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Probabilities", "Confidence", "Attention"])

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
                st.image(base64.b64decode(overlay_b64), use_column_width=True)
                st.markdown(
                    """
                    <div style="margin-top:0.4rem;">
                        <span class="legend-item"><span class="legend-swatch" style="background:#B33A3A;"></span>High attention</span>
                        <span class="legend-item"><span class="legend-swatch" style="background:#D9B23C;"></span>Medium</span>
                        <span class="legend-item"><span class="legend-swatch" style="background:#2C5F82;"></span>Low</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Attention visualization not available")

        with st.expander("Medical Guidance"):
            st.write("""
            **Clinical Recommendations:**
            - Consult with a neurologist for proper diagnosis
            - Consider additional imaging if recommended
            - This AI analysis is for educational purposes
            - Always seek professional medical advice
            """)

else:
    st.info("Upload a brain MRI image above to begin analysis.")

    col_feat1, col_feat2 = st.columns(2)
    with col_feat1:
        st.markdown("**Features**")
        st.write("Grad-CAM attention maps")
        st.write("Confidence analysis")
        st.write("Probability distribution")
        st.write("Uncertainty measurement")

    with col_feat2:
        st.markdown("**Process**")
        st.write("1. Upload MRI image")
        st.write("2. Click analyze")
        st.write("3. View AI results")
        st.write("4. Review the diagnosis")

st.markdown("---")
st.caption("NeuroSight Brain Tumor Classification System · 91.62% Accuracy · Educational Use Only")
