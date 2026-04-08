import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path
from collections import defaultdict

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VehicleScope · AI Counter",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0c10;
    color: #e2e8f0;
  }
  .stApp { background: #0a0c10; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #10131a !important;
    border-right: 1px solid #1e2535;
  }
  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8;
  }

  /* ── Header brand ── */
  .brand-header {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: .25rem;
  }
  .brand-sub {
    font-family: 'Space Mono', monospace;
    font-size: .75rem;
    color: #475569;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 2rem;
  }

  /* ── Metric cards ── */
  .metric-row { display: flex; gap: 1rem; margin: 1.5rem 0; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 130px;
    background: linear-gradient(145deg, #111827, #1a2236);
    border: 1px solid #1e2d45;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
  }
  .metric-card.blue::before  { background: linear-gradient(90deg,#38bdf8,#818cf8); }
  .metric-card.green::before { background: linear-gradient(90deg,#34d399,#059669); }
  .metric-card.pink::before  { background: linear-gradient(90deg,#f472b6,#a855f7); }
  .metric-card.amber::before { background: linear-gradient(90deg,#fbbf24,#f97316); }
  .metric-label {
    font-size: .7rem; letter-spacing: 2px; text-transform: uppercase;
    color: #64748b; margin-bottom: .5rem;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem; font-weight: 700; color: #f1f5f9;
    line-height: 1;
  }
  .metric-delta { font-size: .75rem; color: #38bdf8; margin-top: .25rem; }

  /* ── Upload zone ── */
  div[data-testid="stFileUploader"] {
    background: #10131a;
    border: 2px dashed #1e2d45;
    border-radius: 16px;
    padding: 1rem;
    transition: border-color .2s;
  }
  div[data-testid="stFileUploader"]:hover { border-color: #38bdf8; }

  /* ── Buttons ── */
  .stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: .8rem !important;
    letter-spacing: 1px !important;
    background: linear-gradient(135deg, #1d4ed8, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .65rem 1.6rem !important;
    transition: opacity .2s, transform .1s !important;
  }
  .stButton > button:hover { opacity: .9; transform: translateY(-1px); }
  .stButton > button:active { transform: translateY(0); }

  /* ── Sliders ── */
  div[data-testid="stSlider"] .stSlider > div > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8) !important;
  }

  /* ── Progress bar ── */
  div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6) !important;
  }

  /* ── Info / warning boxes ── */
  div[data-testid="stInfo"]    { background: #0f2744; border-left-color: #38bdf8; border-radius: 10px; }
  div[data-testid="stSuccess"] { background: #052e16; border-left-color: #34d399; border-radius: 10px; }
  div[data-testid="stWarning"] { background: #1c1407; border-left-color: #fbbf24; border-radius: 10px; }

  /* ── Divider ── */
  hr { border-color: #1e2535 !important; }

  /* ── Section label ── */
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: .65rem; letter-spacing: 3px; text-transform: uppercase;
    color: #38bdf8; margin-bottom: 1rem;
  }

  /* ── Status pill ── */
  .status-pill {
    display: inline-flex; align-items: center; gap: .5rem;
    background: #052e16; border: 1px solid #166534;
    color: #34d399; border-radius: 999px;
    padding: .3rem .9rem; font-size: .75rem;
    font-family: 'Space Mono', monospace;
  }
  .status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #34d399;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: .3; }
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_model():
    """Load YOLOv8 model (cached)."""
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")   # nano – fast; swap for yolov8s/m for accuracy
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


VEHICLE_CLASSES = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
CLASS_COLORS = {
    "car":        (56,  189, 248),   # sky-blue
    "truck":      (248, 113, 113),   # red
    "bus":        (167, 139, 250),   # violet
    "motorcycle": (52,  211, 153),   # green
}


def draw_detection_box(frame, x1, y1, x2, y2, label, conf, color):
    """Draw rounded-corner bounding box with label."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    txt = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, txt, (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (10, 12, 20), 1, cv2.LINE_AA)


def process_video(
    video_path: str,
    conf_threshold: float,
    line_position: float,
    selected_classes: list,
    progress_bar,
    status_text,
) -> tuple[str, dict]:
    """
    Run YOLO detection + line-crossing counter on every frame.
    Returns (output_path, counts_dict).
    """
    model = get_model()
    if model is None:
        return None, {}

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    line_y = int(height * line_position)

    # Output file
    out_path = tempfile.mktemp(suffix="_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Counting state
    counts      = defaultdict(int)
    track_prev  = {}           # track_id → prev centre_y
    counted_ids = set()        # ids already counted

    # Class id → name filter
    allowed_ids = {VEHICLE_CLASSES[c] for c in selected_classes if c in VEHICLE_CLASSES}

    frame_idx = 0
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── Run YOLO with tracking ──────────────────────────────────────────
        results = model.track(
            frame,
            conf=conf_threshold,
            classes=list(allowed_ids),
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
        )

        annotated = frame.copy()

        # ── Draw counting line ──────────────────────────────────────────────
        cv2.line(annotated, (0, line_y), (width, line_y), (251, 191, 36), 2)
        cv2.putText(annotated, "COUNTING LINE", (10, line_y - 8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (251, 191, 36), 1, cv2.LINE_AA)

        # ── Process detections ──────────────────────────────────────────────
        if results[0].boxes is not None and len(results[0].boxes):
            boxes  = results[0].boxes
            ids    = boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(boxes)
            clses  = boxes.cls.int().cpu().tolist()
            confs  = boxes.conf.cpu().tolist()
            xyxys  = boxes.xyxy.int().cpu().tolist()

            for tid, cls_id, conf, (x1, y1, x2, y2) in zip(ids, clses, confs, xyxys):
                cls_name = model.names[cls_id]
                color    = CLASS_COLORS.get(cls_name, (200, 200, 200))

                draw_detection_box(annotated, x1, y1, x2, y2, cls_name, conf, color)

                if tid is not None:
                    cy = (y1 + y2) // 2
                    # Draw centre dot
                    cv2.circle(annotated, ((x1+x2)//2, cy), 4, color, -1)

                    # Line-crossing logic
                    prev_cy = track_prev.get(tid)
                    if prev_cy is not None and tid not in counted_ids:
                        crossed = (prev_cy < line_y <= cy) or (prev_cy > line_y >= cy)
                        if crossed:
                            counts[cls_name] += 1
                            counted_ids.add(tid)

                    track_prev[tid] = cy

        # ── HUD overlay ────────────────────────────────────────────────────
        total = sum(counts.values())
        hud_lines = [f"TOTAL: {total}"] + [f"{k.upper()}: {v}" for k, v in counts.items()]
        for i, ln in enumerate(hud_lines):
            y_pos = 28 + i * 22
            cv2.putText(annotated, ln, (width - 170, y_pos),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5,
                        (255, 255, 255) if i == 0 else (180, 180, 200),
                        1, cv2.LINE_AA)

        # Frame counter
        elapsed = time.time() - t0
        fps_live = frame_idx / max(elapsed, 0.001)
        cv2.putText(annotated, f"FPS {fps_live:.1f}  |  FRAME {frame_idx}/{total_frames}",
                    (10, height - 10), cv2.FONT_HERSHEY_DUPLEX, 0.4,
                    (100, 120, 160), 1, cv2.LINE_AA)

        out.write(annotated)

        frame_idx += 1
        if total_frames > 0:
            pct = frame_idx / total_frames
            progress_bar.progress(min(pct, 1.0))
            status_text.markdown(
                f'<div class="status-pill">'
                f'<div class="status-dot"></div>'
                f'Processing frame {frame_idx}/{total_frames} · {pct*100:.1f}%'
                f'</div>',
                unsafe_allow_html=True,
            )

    cap.release()
    out.release()
    return out_path, dict(counts)


# ── UI ─────────────────────────────────────────────────────────────────────────

# Header
st.markdown('<div class="brand-header">VehicleScope</div>', unsafe_allow_html=True)
st.markdown('<div class="brand-sub">AI · Vehicle Detection & Counting</div>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Detection Settings")
    st.markdown("---")

    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.1, max_value=0.95, value=0.40, step=0.05,
        help="Minimum confidence for a detection to be counted.",
    )

    line_pos = st.slider(
        "Counting Line Position",
        min_value=0.2, max_value=0.8, value=0.50, step=0.05,
        help="Vertical position of the counting line as fraction of frame height.",
        format="%.0f%%",
    )
    # convert 0–1 display to fraction
    line_pos_frac = line_pos

    st.markdown("---")
    st.markdown("### 🚗 Vehicle Classes")
    sel_car   = st.checkbox("🚗 Car",        value=True)
    sel_truck = st.checkbox("🚛 Truck",      value=True)
    sel_bus   = st.checkbox("🚌 Bus",        value=True)
    sel_moto  = st.checkbox("🏍️ Motorcycle", value=False)

    selected_classes = []
    if sel_car:   selected_classes.append("car")
    if sel_truck: selected_classes.append("truck")
    if sel_bus:   selected_classes.append("bus")
    if sel_moto:  selected_classes.append("motorcycle")

    st.markdown("---")
    save_output = st.checkbox("💾 Save annotated video", value=True)

    st.markdown("---")
    st.markdown(
        '<div style="font-size:.7rem;color:#334155;font-family:Space Mono,monospace;">'
        'Model: YOLOv8n · Tracker: ByteTrack<br>'
        'Detection: COCO classes'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Main panel ──────────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([3, 2], gap="large")

with col_upload:
    st.markdown('<div class="section-label">📁 Upload Video</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop your video file here",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

with col_info:
    st.markdown('<div class="section-label">ℹ️ How It Works</div>', unsafe_allow_html=True)
    st.info(
        "**1.** Upload a video (mp4 / avi / mov)  \n"
        "**2.** Adjust confidence & line position  \n"
        "**3.** Click **Run Detection**  \n"
        "**4.** Download the annotated result"
    )

# ── Metrics placeholder ─────────────────────────────────────────────────────────
metrics_placeholder = st.empty()

# ── Process ─────────────────────────────────────────────────────────────────────
if uploaded:
    st.markdown("---")

    # Preview
    st.markdown('<div class="section-label">🎬 Uploaded Video Preview</div>', unsafe_allow_html=True)
    st.video(uploaded)

    st.markdown("---")

    run_btn = st.button("▶ Run Vehicle Detection", use_container_width=True)

    if run_btn:
        if not selected_classes:
            st.warning("Please select at least one vehicle class in the sidebar.")
        else:
            # Save upload to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            st.markdown("---")
            st.markdown('<div class="section-label">⚡ Processing</div>', unsafe_allow_html=True)
            prog = st.progress(0)
            status = st.empty()

            t_start = time.time()
            out_path, counts = process_video(
                tmp_path,
                conf_threshold=conf_thresh,
                line_position=line_pos_frac,
                selected_classes=selected_classes,
                progress_bar=prog,
                status_text=status,
            )
            elapsed = time.time() - t_start

            # Clean up temp input
            os.unlink(tmp_path)

            prog.progress(1.0)
            status.empty()

            if out_path and os.path.exists(out_path):
                total_count = sum(counts.values())

                # ── Metric cards ───────────────────────────────────────────
                st.markdown("---")
                st.markdown('<div class="section-label">📊 Results</div>', unsafe_allow_html=True)

                card_colors = ["blue", "green", "pink", "amber"]
                card_icons  = ["🚗", "🚛", "🚌", "🏍️"]
                card_keys   = ["car", "truck", "bus", "motorcycle"]

                total_html = f"""
                <div class="metric-row">
                  <div class="metric-card blue" style="min-width:200px">
                    <div class="metric-label">Total Vehicles Crossed</div>
                    <div class="metric-value">{total_count}</div>
                    <div class="metric-delta">⏱ {elapsed:.1f}s processing time</div>
                  </div>
                """
                for icon, key, clr in zip(card_icons, card_keys, card_colors):
                    v = counts.get(key, 0)
                    if v > 0 or key in selected_classes:
                        total_html += f"""
                  <div class="metric-card {clr}">
                    <div class="metric-label">{icon} {key.capitalize()}s</div>
                    <div class="metric-value">{v}</div>
                  </div>"""
                total_html += "</div>"

                metrics_placeholder.markdown(total_html, unsafe_allow_html=True)

                # ── Annotated video ────────────────────────────────────────
                st.markdown('<div class="section-label">🎬 Annotated Output</div>', unsafe_allow_html=True)
                st.video(out_path)

                # ── Download ───────────────────────────────────────────────
                if save_output:
                    with open(out_path, "rb") as f:
                        video_bytes = f.read()
                    st.download_button(
                        label="⬇ Download Annotated Video",
                        data=video_bytes,
                        file_name="vehiclescope_output.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

                st.success(
                    f"✅ Detection complete · **{total_count}** vehicle(s) counted "
                    f"crossing the line in {elapsed:.1f}s"
                )

                # Clean up temp output
                try:
                    os.unlink(out_path)
                except Exception:
                    pass
            else:
                st.error("Processing failed. Please check the model installation and try again.")

else:
    # Landing state metrics (zeros)
    metrics_placeholder.markdown("""
    <div class="metric-row">
      <div class="metric-card blue">
        <div class="metric-label">Total Vehicles</div>
        <div class="metric-value">—</div>
        <div class="metric-delta">Upload a video to begin</div>
      </div>
      <div class="metric-card green">
        <div class="metric-label">🚗 Cars</div>
        <div class="metric-value">—</div>
      </div>
      <div class="metric-card pink">
        <div class="metric-label">🚌 Buses</div>
        <div class="metric-value">—</div>
      </div>
      <div class="metric-card amber">
        <div class="metric-label">🚛 Trucks</div>
        <div class="metric-value">—</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("👆 Upload a video in the panel above to get started.")
