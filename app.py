import os
import cv2
import altair as alt
import pandas as pd
import streamlit as st
from detector import PotholeDetector
from report_generator import ReportGenerator

# -------------------------------------------------------------------
# Page Configuration & Interactive Dashboard Styling
# -------------------------------------------------------------------
st.set_page_config(
    page_title="PaveVision AI - Pothole Detection System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main Banner */
    .main-header {
        background: linear-gradient(135deg, #0b1120 0%, #1e293b 60%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        border: 1px solid #334155;
        border-left: 6px solid #38bdf8;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-header h1 {
        color: #f8fafc;
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .main-header p {
        color: #94a3b8;
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
    }
    .status-pill {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Interactive Severity Legend Cards */
    .severity-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.3rem;
    }
    .sev-card {
        background: #1e293b;
        padding: 0.85rem 1.1rem;
        border-radius: 10px;
        border: 1px solid #334155;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .sev-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
    }
    .sev-high { border-left: 5px solid #ef4444; }
    .sev-med  { border-left: 5px solid #facc15; }
    .sev-low  { border-left: 5px solid #22c55e; }

    .sev-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .sev-desc {
        font-size: 0.82rem;
        color: #94a3b8;
    }
    </style>
""", unsafe_allow_html=True)

# Ensure working directories exist
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Initialize Session State so downloads & filters don't reset the dashboard
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "results_data" not in st.session_state:
    st.session_state.results_data = {}

# -------------------------------------------------------------------
# Header & Interactive Severity Cards
# -------------------------------------------------------------------
st.markdown("""
    <div class="main-header">
        <div>
            <h1>🛣️ PaveVision AI - Pothole Detection System</h1>
            <p>Automated Road Surface Defect Inspection, Severity Grading & Engineering Report Generator</p>
        </div>
        <div class="status-pill">● AI ENGINE READY</div>
    </div>

    <div class="severity-grid">
        <div class="sev-card sev-high">
            <div class="sev-title">🔴 Deep / High Severity</div>
            <div class="sev-desc">Critical structural damage • Immediate patching required</div>
        </div>
        <div class="sev-card sev-med">
            <div class="sev-title">🟡 Medium Level</div>
            <div class="sev-desc">Moderate surface depression • Scheduled maintenance</div>
        </div>
        <div class="sev-card sev-low">
            <div class="sev-title">🟢 Little / Low Severity</div>
            <div class="sev-desc">Minor surface wear • Routine monitoring</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Sidebar Inspection Controls
# -------------------------------------------------------------------
st.sidebar.title("🎛️ Inspection Controls")
road_name = st.sidebar.text_input(
    "Road / Highway Stretch",
    value="Indore - Bhopal Highway Stretch"
)
conf_thresh = st.sidebar.slider(
    "Detection Sensitivity",
    min_value=0.20,
    max_value=0.80,
    value=0.40,
    step=0.05
)
model_path = st.sidebar.text_input(
    "YOLOv8 Model Path",
    value="models/pothole_best.pt"
)

st.sidebar.divider()
st.sidebar.info("💡 **Tip:** Upload a clear road image or dashcam video, then click **Start Pothole Detection**.")

# Initialize Detector
detector = PotholeDetector(model_path=model_path)

# -------------------------------------------------------------------
# File Upload & Execution Trigger
# -------------------------------------------------------------------
up_col1, up_col2 = st.columns([2.3, 1])

with up_col1:
    uploaded_file = st.file_uploader(
        "Upload Road Image or Dashcam Video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )

with up_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button(
        "🚀 Start Pothole Detection",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None)
    )

if uploaded_file is not None and not run_btn and not st.session_state.analysis_done:
    file_size_mb = round(uploaded_file.size / (1024 * 1024), 2)
    st.caption(f"📎 Selected File: **{uploaded_file.name}** ({file_size_mb} MB) — Ready for inspection.")

# -------------------------------------------------------------------
# Run Detection ONLY when button is clicked & save to Session State
# -------------------------------------------------------------------
if uploaded_file is not None and run_btn:
    input_path = os.path.join("temp_uploads", uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_ext = uploaded_file.name.split(".")[-1].lower()
    is_image = file_ext in ["jpg", "jpeg", "png"]
    base_name = os.path.splitext(uploaded_file.name)[0]

    pdf_path = os.path.join("outputs", f"{base_name}_Report.pdf")
    csv_path = os.path.join("outputs", f"{base_name}_Data.csv")

    if is_image:
        with st.spinner("Scanning road surface for potholes..."):
            out_media_path = os.path.join("outputs", f"detected_{uploaded_file.name}")
            out_media_path, detections = detector.process_image(
                input_path, out_media_path, conf_thresh
            )
            preview_img = out_media_path
    else:
        progress_bar = st.progress(0)
        live_status = st.empty()

        # Average balanced live preview container
        _, mid_col, _ = st.columns([0.6, 2, 0.6])
        with mid_col:
            live_frame_box = st.empty()

        def update_live_preview(frame_bgr, current_f, total_f, count_so_far):
            pct = min(int((current_f / total_f) * 100), 100)
            progress_bar.progress(pct)
            live_status.markdown(
                f"⏳ **Scanning Frame:** `{current_f}/{total_f}` &nbsp;|&nbsp; "
                f"**Unique Potholes Detected:** `{count_so_far}`"
            )
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            live_frame_box.image(frame_rgb, channels="RGB", width=640)

        out_media_path = os.path.join("outputs", f"detected_{base_name}.mp4")
        out_media_path, detections = detector.process_video(
            input_path,
            out_media_path,
            conf_thresh,
            progress_callback=update_live_preview
        )

        progress_bar.progress(100)
        live_status.empty()
        live_frame_box.empty()
        preview_img = None

    # Generate PDF and CSV Reports
    ReportGenerator.generate_pdf(
        detections, road_name, uploaded_file.name, pdf_path, preview_img
    )
    ReportGenerator.generate_csv(detections, csv_path)

    # Save everything in Session State so it persists across downloads & tab clicks
    st.session_state.analysis_done = True
    st.session_state.results_data = {
        "detections": detections,
        "is_image": is_image,
        "out_media_path": out_media_path,
        "pdf_path": pdf_path,
        "csv_path": csv_path,
        "road_name": road_name
    }

# -------------------------------------------------------------------
# Render Persistent Dashboard Analytics (Stays visible on Download!)
# -------------------------------------------------------------------
if st.session_state.analysis_done and st.session_state.results_data:
    res = st.session_state.results_data
    detections = res["detections"]
    is_image = res["is_image"]
    out_media_path = res["out_media_path"]
    pdf_path = res["pdf_path"]
    csv_path = res["csv_path"]
    saved_road_name = res["road_name"]

    # Calculate Metrics
    total = len(detections)
    high = sum(1 for d in detections if d["severity"] == "High")
    med = sum(1 for d in detections if d["severity"] == "Medium")
    low = sum(1 for d in detections if d["severity"] == "Low")
    total_area = round(sum(d["est_area_sqm"] for d in detections), 2)

    # KPI Summary Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Potholes", total)
    m2.metric("🔴 Deep / High", high)
    m3.metric("🟡 Medium Level", med)
    m4.metric("🟢 Little / Low", low)
    m5.metric("Est. Patch Area", f"{total_area} m²")

    st.divider()

    # Tabbed Inspection Dashboard
    tab1, tab2, tab3 = st.tabs([
        "🎥 Visual Inspection Output",
        "📊 Interactive Pothole Log",
        "📥 Export Engineering Reports"
    ])

    with tab1:
        v_col1, v_col2 = st.columns([1.65, 1])

        with v_col1:
            st.subheader("Annotated Road Feed")
            if is_image and os.path.exists(out_media_path):
                st.image(out_media_path, width=680)
            elif not is_image and os.path.exists(out_media_path):
                with open(out_media_path, "rb") as vf:
                    st.video(vf.read())

        with v_col2:
            st.subheader("Road Condition Assessment")
            if total == 0:
                st.success("Road surface is clear. No major potholes detected.")
            elif high > 0:
                st.error(
                    f"⚠️ Critical Maintenance Required: {high} deep pothole(s) "
                    f"detected on {saved_road_name}."
                )
            else:
                st.warning(
                    f"Moderate surface wear detected ({total} potholes). "
                    f"Schedule routine patching."
                )

            # Color-Coded Interactive Bar Chart (Red = High, Yellow = Medium, Green = Low)
            chart_df = pd.DataFrame({
                "Severity": ["High (Deep)", "Medium", "Low (Little)"],
                "Count": [high, med, low]
            })

            color_scale = alt.Scale(
                domain=["High (Deep)", "Medium", "Low (Little)"],
                range=["#ef4444", "#facc15", "#22c55e"]
            )

            bar_chart = (
                alt.Chart(chart_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=55)
                .encode(
                    x=alt.X(
                        "Severity:N",
                        sort=["High (Deep)", "Medium", "Low (Little)"],
                        axis=alt.Axis(labelAngle=0, title=None)
                    ),
                    y=alt.Y("Count:Q", axis=alt.Axis(title="Pothole Count")),
                    color=alt.Color("Severity:N", scale=color_scale, legend=None),
                    tooltip=["Severity", "Count"]
                )
                .properties(height=260)
            )

            st.altair_chart(bar_chart, use_container_width=True)

    with tab2:
        st.subheader("Detected Potholes Registry")
        df = pd.DataFrame(detections)
        if not df.empty:
            selected_sev = st.multiselect(
                "Filter by Severity Level:",
                options=["High", "Medium", "Low"],
                default=["High", "Medium", "Low"]
            )
            filtered_df = df[df["severity"].isin(selected_sev)]
            st.dataframe(filtered_df, use_container_width=True, height=340)
        else:
            st.info("No potholes recorded at the current sensitivity threshold.")

    with tab3:
        st.subheader("Download Official Reports & Media")
        d1, d2, d3 = st.columns(3)

        with d1:
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "📄 Download PDF Report",
                        data=f.read(),
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        use_container_width=True,
                        key="dl_pdf"
                    )

        with d2:
            if os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button(
                        "📊 Download CSV Data",
                        data=f.read(),
                        file_name=os.path.basename(csv_path),
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_csv"
                    )

        if not is_image and os.path.exists(out_media_path):
            with d3:
                with open(out_media_path, "rb") as f:
                    st.download_button(
                        "🎬 Download Annotated Video",
                        data=f.read(),
                        file_name=os.path.basename(out_media_path),
                        mime="video/mp4",
                        use_container_width=True,
                        key="dl_vid"
                    )