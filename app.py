from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image
import streamlit as st

import model as model_utils


CONFIDENCE_THRESHOLD = 0.35
BASE_DIR = Path(__file__).parent


@st.cache_resource(show_spinner="Loading wildfire detection model...")
def load_detector():
    weights = model_utils.find_weights(BASE_DIR)
    return model_utils.load_model(weights)


def compute_risk(detections, names):
    if not detections:
        return {
            "decision": "No fire or smoke detected",
            "risk": "Safe",
            "message": "No fire or smoke signals were detected in the uploaded media.",
            "fire_count": 0,
            "smoke_count": 0,
            "max_confidence": 0.0,
        }

    fire_count = 0
    smoke_count = 0
    max_confidence = 0.0

    for detection in detections:
        max_confidence = max(max_confidence, detection["conf"])
        label = str(names.get(detection["cls"], detection["cls"]) if isinstance(names, dict) else names[detection["cls"]])
        label = label.lower()
        if "fire" in label:
            fire_count += 1
        elif "smoke" in label:
            smoke_count += 1

    if fire_count and smoke_count:
        decision = "Fire and smoke detected"
    elif fire_count:
        decision = "Fire detected"
    elif smoke_count:
        decision = "Smoke detected"
    else:
        decision = "Object detected"

    if fire_count and max_confidence >= 0.75:
        risk = "High"
        message = "Strong wildfire indicators detected. Immediate review is recommended."
    elif fire_count or (smoke_count and max_confidence >= 0.55):
        risk = "Medium"
        message = "Possible fire or smoke activity detected. Monitor the area closely."
    elif smoke_count:
        risk = "Low"
        message = "Low-confidence smoke-like signals detected. Visual verification is advised."
    else:
        risk = "Low"
        message = "Objects were detected, but not clearly fire or smoke."

    return {
        "decision": decision,
        "risk": risk,
        "message": message,
        "fire_count": fire_count,
        "smoke_count": smoke_count,
        "max_confidence": float(max_confidence),
    }


def bgr_to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def analyze_image(image_bgr, model):
    results = model_utils.predict_image(model, image_bgr, conf=CONFIDENCE_THRESHOLD)
    detections = model_utils.parse_detections(results, conf_thresh=CONFIDENCE_THRESHOLD)
    annotated = model_utils.annotate_image_np(image_bgr, detections)
    summary = compute_risk(detections, getattr(model, "names", {}))
    return annotated, detections, summary


def draw_live_overlay(image_bgr, summary):
    colors = {
        "Safe": (22, 163, 74),
        "Low": (202, 138, 4),
        "Medium": (234, 88, 12),
        "High": (220, 38, 38),
    }
    color = colors.get(summary["risk"], (37, 99, 235))
    label = f"{summary['decision']} | Risk: {summary['risk']} | Conf: {summary.get('max_confidence', 0.0):.0%}"

    output = image_bgr.copy()
    cv2.rectangle(output, (10, 10), (min(output.shape[1] - 10, 760), 58), (15, 23, 42), -1)
    cv2.putText(output, label, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)
    return output


def analyze_video(uploaded_file, model, frame_stride=10, max_frames=80):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_file:
        input_file.write(uploaded_file.getvalue())
        input_path = Path(input_file.name)

    output_path = input_path.with_name(f"{input_path.stem}_annotated.mp4")
    detections_seen = []
    frames_processed = 0
    frames_with_alerts = 0
    preview_frame = None

    try:
        capture = cv2.VideoCapture(str(input_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 24
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame_index = 0
        while capture.isOpened() and frames_processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % frame_stride == 0:
                annotated, detections, _ = analyze_image(frame, model)
                detections_seen.extend(detections)
                if detections:
                    frames_with_alerts += 1
                writer.write(annotated)
                preview_frame = annotated
                frames_processed += 1
            frame_index += 1

        capture.release()
        writer.release()

        summary = compute_risk(detections_seen, getattr(model, "names", {}))
        summary["frames_processed"] = frames_processed
        summary["frames_with_alerts"] = frames_with_alerts
        return output_path, preview_frame, summary
    finally:
        input_path.unlink(missing_ok=True)


def inject_css():
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #fff7ed 0%, #fef2f2 52%, #f8fafc 100%);
                color: #111827;
            }
            .block-container {
                max-width: 1180px;
                padding-top: 2rem;
            }
            .hero {
                background:
                    radial-gradient(circle at top right, rgba(250, 204, 21, 0.42), transparent 22rem),
                    linear-gradient(135deg, #7f1d1d 0%, #dc2626 58%, #f97316 100%);
                color: white;
                padding: 30px;
                border-radius: 16px;
                box-shadow: 0 22px 55px rgba(220, 38, 38, 0.25);
                margin-bottom: 20px;
            }
            .hero h1 {
                color: white;
                margin: 0;
                font-size: 42px;
                line-height: 1.05;
            }
            .hero p {
                color: #fff7ed;
                font-size: 17px;
                max-width: 780px;
                margin-top: 10px;
            }
            .risk-card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(248, 113, 113, 0.25);
                border-radius: 14px;
                padding: 18px;
                box-shadow: 0 12px 32px rgba(127, 29, 29, 0.10);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_summary(summary):
    colors = {
        "Safe": "#16a34a",
        "Low": "#ca8a04",
        "Medium": "#ea580c",
        "High": "#dc2626",
    }
    color = colors.get(summary["risk"], "#2563eb")
    st.markdown(
        f"""
        <div class="risk-card">
            <div style="color:#64748b;font-size:14px;">Detection Summary</div>
            <div style="font-size:32px;font-weight:800;color:{color};">{summary["decision"]}</div>
            <div style="font-size:18px;color:#334155;margin-top:4px;">Risk Level: <b>{summary["risk"]}</b></div>
            <div style="color:#475569;margin-top:8px;">{summary["message"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Fire Detections", summary.get("fire_count", 0))
    metric_2.metric("Smoke Detections", summary.get("smoke_count", 0))
    metric_3.metric("Max Confidence", f"{summary.get('max_confidence', 0.0):.1%}")
    metric_4.metric("Risk", summary["risk"])


def image_page(model):
    st.subheader("Image Detection")
    uploaded_image = st.file_uploader(
        "Upload a forest/fire image",
        type=["jpg", "jpeg", "png", "webp", "jfif"],
    )

    if uploaded_image and st.button("Analyze Image", type="primary", use_container_width=True):
        image_bgr = read_image(uploaded_image)
        annotated, detections, summary = analyze_image(image_bgr, model)
        render_summary(summary)

        original_col, result_col = st.columns(2)
        with original_col:
            st.caption("Original Image")
            st.image(bgr_to_rgb(image_bgr), use_container_width=True)
        with result_col:
            st.caption("Annotated Detection")
            st.image(bgr_to_rgb(annotated), use_container_width=True)

        if detections:
            st.subheader("Raw Detections")
            st.dataframe(detections, use_container_width=True)


def video_page(model):
    st.subheader("Video Detection")
    st.info("For cloud performance, the app samples video frames instead of processing every frame.")
    uploaded_video = st.file_uploader("Upload a short video", type=["mp4", "avi", "mov", "mkv"])
    frame_stride = st.slider("Analyze every Nth frame", min_value=3, max_value=30, value=10)
    max_frames = st.slider("Maximum analyzed frames", min_value=20, max_value=160, value=80)

    if uploaded_video and st.button("Analyze Video", type="primary", use_container_width=True):
        with st.spinner("Processing sampled video frames..."):
            output_path, preview_frame, summary = analyze_video(uploaded_video, model, frame_stride, max_frames)

        render_summary(summary)
        st.metric("Frames Processed", summary.get("frames_processed", 0))
        st.metric("Frames With Alerts", summary.get("frames_with_alerts", 0))

        if preview_frame is not None:
            st.caption("Latest Annotated Frame")
            st.image(bgr_to_rgb(preview_frame), use_container_width=True)

        if output_path.exists():
            st.download_button(
                "Download Annotated Video",
                data=output_path.read_bytes(),
                file_name="wildfire_detection_annotated.mp4",
                mime="video/mp4",
                use_container_width=True,
            )
            output_path.unlink(missing_ok=True)


def live_camera_page(model):
    st.subheader("Real-Time Live Fire Detection")
    st.write(
        "Use your browser camera for live wildfire monitoring. The app analyzes incoming "
        "frames and overlays fire/smoke risk directly on the video feed."
    )

    st.warning(
        "Cloud note: live camera works through browser WebRTC. If your browser asks for "
        "camera permission, allow it. For slow devices, increase frame skipping."
    )

    frame_skip = st.slider(
        "Analyze every Nth live frame",
        min_value=1,
        max_value=10,
        value=3,
        help="Higher values reduce CPU load on Streamlit Cloud.",
    )

    try:
        import av
        from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
    except ImportError:
        st.error("Live WebRTC dependencies are not installed. Using snapshot fallback.")
        live_snapshot_fallback(model)
        return

    class WildfireVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.frame_count = 0
            self.last_frame = None

        def recv(self, frame):
            image_bgr = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            if self.frame_count % frame_skip == 0 or self.last_frame is None:
                annotated, _, summary = analyze_image(image_bgr, self.model)
                self.last_frame = draw_live_overlay(annotated, summary)

            return av.VideoFrame.from_ndarray(self.last_frame, format="bgr24")

    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.info("Click START below to begin live camera detection.")
    webrtc_streamer(
        key="wildfire-live-camera",
        video_processor_factory=WildfireVideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    with st.expander("Fallback: Analyze Camera Snapshot"):
        live_snapshot_fallback(model)


def live_snapshot_fallback(model):
    snapshot = st.camera_input("Take a camera snapshot")
    if snapshot is not None:
        image_bgr = read_image(snapshot)
        annotated, _, summary = analyze_image(image_bgr, model)
        render_summary(summary)
        st.image(bgr_to_rgb(annotated), caption="Detected Snapshot", use_container_width=True)


def about_page():
    st.subheader("About This Project")
    st.write(
        "This Streamlit app uses a YOLO object detection model to identify wildfire "
        "signals such as fire and smoke in uploaded images, sampled video frames, "
        "and live browser camera feed."
    )
    st.write(
        "The system is intended as a demonstration and early-warning support tool. "
        "Real-world wildfire response should always involve human verification and official emergency channels."
    )


def main():
    st.set_page_config(page_title="Wildfire Detection", page_icon="fire", layout="wide")
    inject_css()
    st.markdown(
        """
        <section class="hero">
            <h1>Wildfire Detection System</h1>
            <p>
                Streamlit dashboard for detecting fire and smoke in uploaded images and videos
                using a YOLO-based object detection model.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Wildfire AI")
        page = st.radio("Navigation", ["Image Detection", "Video Detection", "Live Camera Detection", "About"])
        st.divider()
        st.metric("Confidence Threshold", f"{CONFIDENCE_THRESHOLD:.0%}")
        st.caption("Model file: best.pt / last.pt")

    model = load_detector()

    if page == "Image Detection":
        image_page(model)
    elif page == "Video Detection":
        video_page(model)
    elif page == "Live Camera Detection":
        live_camera_page(model)
    else:
        about_page()


if __name__ == "__main__":
    main()
