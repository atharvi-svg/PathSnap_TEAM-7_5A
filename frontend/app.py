from __future__ import annotations

import html
import os

import requests
import streamlit as st
import streamlit.components.v1 as components

from frontend.map_view import route_map_html

API_URL = os.getenv("PATHSNAP_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="PathSnap | Route Explorer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: #f4f7f6 !important;
        color: #18332d !important;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stToolbar"] { right: 1rem; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1560px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2eae7;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
    [data-testid="stSidebar"] * { color: #29443d; }
    [data-testid="stMarkdownContainer"] p { color: #5c716b; }
    h1, h2, h3 { color: #173c32 !important; letter-spacing: -0.025em; }
    h1 { font-size: 2rem !important; font-weight: 750 !important; margin-bottom: .15rem !important; }
    h2 { font-size: 1.2rem !important; font-weight: 700 !important; }
    h3 { font-size: 1rem !important; font-weight: 700 !important; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1eae6;
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 3px 14px rgba(20, 61, 49, .035);
    }
    [data-testid="stMetricLabel"] { color: #6d807a !important; font-size: .8rem !important; }
    [data-testid="stMetricValue"] { color: #183f34 !important; font-weight: 720 !important; }
    [data-testid="stBaseButton-primary"] {
        background: #146b57 !important;
        border: 1px solid #146b57 !important;
        border-radius: 11px !important;
        color: white !important;
        font-weight: 650 !important;
        min-height: 2.8rem;
        box-shadow: 0 4px 12px rgba(20, 107, 87, .16);
    }
    [data-testid="stBaseButton-primary"]:hover { background: #0f5948 !important; }
    [data-testid="stBaseButton-primary"] p { color: #ffffff !important; }
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] > div:first-child { border-color: #146b57 !important; }
    [data-testid="stSelectbox"] > div > div { border-radius: 11px; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #fff;
        border: 1px solid #e1eae6;
        border-radius: 18px;
        box-shadow: 0 6px 20px rgba(20, 61, 49, .035);
    }
    [data-testid="stAlert"] { border-radius: 12px; }
    .brand-mark {
        width: 42px; height: 42px; border-radius: 14px;
        display: inline-flex; align-items: center; justify-content: center;
        color: #fff; background: linear-gradient(145deg, #1a8068, #125a4a);
        font-size: 1.25rem; font-weight: 800; margin-right: 10px;
        box-shadow: 0 5px 14px rgba(20, 107, 87, .2);
        vertical-align: middle;
    }
    .brand-name { color: #183b32; font-size: 1.28rem; font-weight: 800; vertical-align: middle; }
    .brand-sub { color: #8a9b95; font-size: .75rem; margin: 6px 0 22px 52px; }
    .eyebrow { color: #188268; font-size: .72rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
    .page-subtitle { color: #71847e; margin-top: 0; font-size: .98rem; }
    .top-chip {
        display: inline-block; background: #e5f4ee; color: #17684f;
        padding: 7px 11px; border-radius: 999px; font-size: .75rem;
        font-weight: 750; letter-spacing: .035em; white-space: nowrap;
    }
    .map-heading { display:flex; align-items:center; justify-content:space-between; margin: 2px 2px 10px; }
    .map-heading strong { color:#193b32; font-size:1.02rem; }
    .map-heading span { color:#82938d; font-size:.78rem; }
    .insight-card {
        background: #f7faf8; border: 1px solid #e6eeea; border-radius: 14px;
        padding: 14px; margin: 9px 0 12px;
    }
    .insight-title { color:#1b493c; font-size:.86rem; font-weight:750; margin-bottom:5px; }
    .insight-copy { color:#71847e; font-size:.78rem; line-height:1.5; }
    .status-dot { display:inline-block;width:8px;height:8px;border-radius:50%;background:#28a879;margin-right:7px; }
    .sidebar-note { background:#f1f7f4;border:1px solid #e0eee8;border-radius:14px;padding:13px;margin-top:14px; }
    .sidebar-note b { display:block;color:#1c5948;font-size:.8rem;margin-bottom:4px; }
    .sidebar-note span { color:#71847e;font-size:.74rem;line-height:1.45; }
    div[data-testid="stExpander"] { background:#fff;border:1px solid #e1eae6;border-radius:14px; }
    @media (max-width: 900px) {
        [data-testid="stMainBlockContainer"] { padding-left: .7rem !important; padding-right: .7rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.25rem !important; white-space: nowrap; }
        [data-testid="stMetricLabel"] { font-size: .68rem !important; }
        [data-testid="stHorizontalBlock"] { gap: .45rem !important; }
        [data-testid="stBaseButton-primary"] { font-size: .76rem !important; padding-left: .35rem !important; padding-right: .35rem !important; }
        .page-subtitle { font-size: .87rem; }
        .map-heading span { font-size: .68rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div><span class="brand-mark">P</span><span class="brand-name">PathSnap</span></div>'
        '<div class="brand-sub">ROUTE INTELLIGENCE STUDIO</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">Workspace</div>', unsafe_allow_html=True)
    view = st.radio("Workspace", ["Route explorer", "Project overview"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="eyebrow">Current dataset</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note"><b><span class="status-dot"></span>API connected</b>'
        '<span>Predictions run locally through the PathSnap backend.</span></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"API · {API_URL}")
    st.markdown("---")
    st.caption("PathSnap · Map matching & route reconstruction")

if view == "Project overview":
    st.markdown('<div class="eyebrow">Project workspace</div>', unsafe_allow_html=True)
    st.title("PathSnap overview")
    st.markdown('<div class="page-subtitle">A quick look at the route-matching demo and its local services.</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Service", "Connected")
    c2.metric("Prediction mode", "Local API")
    c3.metric("Current partition", os.getenv("PATHSNAP_TRIP_PARTITION", "student_test"))
    st.info("Use Route explorer to choose a trip and view the predicted route. Student-test scores are development diagnostics based on weak route-proximity labels.")
    st.stop()

st.markdown('<div class="eyebrow">Route intelligence</div>', unsafe_allow_html=True)
header_left, header_right = st.columns([5, 1.35], vertical_alignment="center")
with header_left:
    st.title("Route explorer")
    st.markdown('<div class="page-subtitle">Explore GPS traces and network-aware route predictions.</div>', unsafe_allow_html=True)
with header_right:
    st.markdown('<span class="top-chip">LOCAL WORKSPACE</span>', unsafe_allow_html=True)

try:
    health_response = requests.get(f"{API_URL}/health", timeout=5)
    health_response.raise_for_status()
    health = health_response.json()
    trips_response = requests.get(f"{API_URL}/api/v1/trips", timeout=30)
    trips_response.raise_for_status()
    trips = trips_response.json()
except requests.RequestException as exc:
    st.error(f"Cannot reach the PathSnap API at {API_URL}. Start the backend and check the URL. Details: {exc}")
    st.stop()

if not trips:
    st.warning("The configured partition has no trips.")
    st.stop()

trip_ids = [x["trip_id"] for x in trips]
by_id = {x["trip_id"]: x for x in trips}
selected = st.session_state.get("selected_trip", trip_ids[0])
if selected not in by_id:
    selected = trip_ids[0]

st.write("")
selector, action, info = st.columns([3.0, 1.6, 1.8], vertical_alignment="bottom")
with selector:
    selected = st.selectbox(
        "Select a trip",
        trip_ids,
        index=trip_ids.index(selected),
        format_func=lambda tid: f"Trip {tid}  ·  {by_id[tid]['observations']:,} GPS observations",
        key="trip_selector",
    )
    st.session_state["selected_trip"] = selected
with action:
    run_prediction = st.button("✦  Predict route", type="primary", use_container_width=True)
with info:
    st.markdown(
        f'<div style="padding:9px 4px;color:#81928c;font-size:.82rem">'
        f'<span class="status-dot"></span>{html.escape(str(health.get("partition", "dataset")))} · '
        f'{len(trips)} trips available</div>',
        unsafe_allow_html=True,
    )

if run_prediction:
    with st.spinner("Matching GPS observations to the road network…"):
        try:
            response = requests.post(
                f"{API_URL}/api/v1/predict",
                json={"trip_id": selected},
                timeout=300,
            )
            response.raise_for_status()
            st.session_state["last_result"] = response.json()
        except requests.RequestException as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            st.error(f"Route prediction failed: {detail}")

result = st.session_state.get("last_result")
if result and result.get("trip_id") != selected:
    result = None

if not result:
    st.info("Choose a trip and run a prediction to see its map and route diagnostics.")
    st.stop()

metrics = result.get("metrics") or {}
dev_metrics = bool(metrics)

if dev_metrics:
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("HMM match", f"{float(metrics.get('hmm_top1_accuracy', 0)):.1%}")
    m2.metric("Route continuity", f"{float(metrics.get('hmm_route_continuity', 0)):.1%}")
    m3.metric("Candidate misses", f"{float(metrics.get('candidate_miss_rate', 0)):.1%}")
    m4.metric("GPS fixes", f"{int(result['observation_count']):,}")
else:
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("Prediction", "Ready")
    m2.metric("Route segments", f"{len(result['predicted_arc_ids']):,}")
    m3.metric("GPS fixes", f"{int(result['observation_count']):,}")
    m4.metric("Reference labels", "Not used")

st.write("")
with st.container(border=True):
    st.markdown(
        f'<div class="map-heading"><strong>Trip {html.escape(str(result["trip_id"]))}</strong>'
        '<span>GPS observations · predicted HMM route · drag to explore</span></div>',
        unsafe_allow_html=True,
    )
    html_map = route_map_html(
        result["gps_coordinates"],
        result["predicted_route_coordinates"],
        trip_id=result["trip_id"],
    )
    components.html(html_map, height=690, scrolling=False)

if dev_metrics:
    st.caption("Development scores use weak route-proximity labels; they are not independently verified ground truth.")
else:
    st.caption("Evaluator mode: ground-truth labels are not accessed or returned.")

with st.expander("Route details"):
    d1, d2, d3 = st.columns(3)
    d1.metric("Trip ID", result["trip_id"])
    d2.metric("Observation fixes", f"{int(result['observation_count']):,}")
    d3.metric("Predicted arcs", f"{len(result['predicted_arc_ids']):,}")
    if dev_metrics:
        st.json(metrics)
