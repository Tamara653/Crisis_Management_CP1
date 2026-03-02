# =================================================================
#  app.py  —  Fully patched version for improved core.py integration
#  to run: python -m streamlit run .\Model\app.py
# =================================================================

import time
from dataclasses import asdict
import hashlib
import json
import io
import os
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from model.core import Params, init_state, step_once
from viz.figures import sna_figure, timeseries_figure, trust_timeseries_figure


# =================================================================
# Streamlit config
# =================================================================
st.set_page_config(page_title="Crowd ABM Dashboard", layout="wide")
st.title("Toy Model: H‑AI Crowd Management ABM — Network + Time Series")


# =================================================================
# Helpers
# =================================================================

def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def get_data_editor():
    return getattr(st, "data_editor", getattr(st, "experimental_data_editor", None))


def apply_role_loa_to_params(p: Dict[str, Any], defaults_per_role: Dict[str, Any], roles: Dict[str, str]) -> None:
    """Map LoA selection to Params fields."""
    # A1
    loa1 = roles['A1']
    p['A1_reliability'] = float(defaults_per_role["Sensor"][loa1]["reliability"])
    p['A1_latency']     = int(defaults_per_role["Sensor"][loa1]["latency"])

    # A2
    loa2 = roles['A2']
    p['A2_reliability'] = float(defaults_per_role["Analyst"][loa2]["reliability"])
    p['A2_latency']     = int(defaults_per_role["Analyst"][loa2]["latency"])

    # A3
    loa3 = roles['A3']
    p['A3_reliability'] = float(defaults_per_role["Comms"][loa3]["reliability"])
    p['A3_latency']     = int(defaults_per_role["Comms"][loa3]["latency"])

    # A4
    loa4 = roles['A4']
    a4_rel = float(defaults_per_role["Decider"][loa4]["reliability"])
    a4_lat = int(defaults_per_role["Decider"][loa4]["latency"])
    p['A4_reliability']     = a4_rel
    p['A4_latency']             = a4_lat  # Now used by core


def init_or_reinit_state(baseline_df=None):
    params_obj = Params(**st.session_state.params)
    st.session_state.state = init_state(params_obj, baseline_df=baseline_df)
    st.session_state.state.threshold = st.session_state.params.get('threshold', 4.0)


def _df_to_defaults(df: pd.DataFrame) -> dict:
    out = {}
    for role, row in df.iterrows():
        out[role] = {
            "Human":     {"reliability": float(row["Human_rel"]),     "latency": int(row["Human_lat"])},
            "Hybrid":    {"reliability": float(row["Hybrid_rel"]),    "latency": int(row["Hybrid_lat"])},
            "Automated": {"reliability": float(row["Automated_rel"]), "latency": int(row["Automated_lat"])},
        }
    return out

# =================================================================
# Session state bootstrapping
# =================================================================
if "params" not in st.session_state:
    st.session_state.params = asdict(Params())

p = st.session_state.params

if "defaults_per_role" not in st.session_state:
    # Logical defaults per‑role × per‑LoA
    st.session_state.defaults_per_role = {
        "Sensor":  {
            "Human":     {"reliability": 0.40, "latency": 3},
            "Hybrid":    {"reliability": 0.80, "latency": 1},
            "Automated": {"reliability": 0.95, "latency": 0},
        },
        "Analyst": {
            "Human":     {"reliability": 0.60, "latency": 2},
            "Hybrid":    {"reliability": 0.95, "latency": 1},
            "Automated": {"reliability": 0.80, "latency": 0},
        },
        "Comms":   {
            "Human":     {"reliability": 0.85, "latency": 1},
            "Hybrid":    {"reliability": 0.90, "latency": 1},
            "Automated": {"reliability": 0.80, "latency": 0},
        },
        "Decider": {
            "Human":     {"reliability": 0.50, "latency": 2},
            "Hybrid":    {"reliability": 0.90, "latency": 1},
            "Automated": {"reliability": 0.80, "latency": 0},
        },
    }

if "roles" not in st.session_state:
    st.session_state.roles = {
        "A1": "Automated",
        "A2": "Hybrid",
        "A3": "Human",
        "A4": "Human"
    }

if "need_reinit" not in st.session_state:
    st.session_state.need_reinit = False

if "baseline_df" not in st.session_state:
    st.session_state.baseline_df = None

# if "last_reinit_fp" not in st.session_state:
#     st.session_state.last_reinit_fp = ""


# ======================================================================================
# Sidebar: Roles & LoA + Trust
# ======================================================================================
st.sidebar.subheader("Agent roles + LoA")

def mark_need_reinit():
    st.session_state.need_reinit = True

# Role selections (persisted with keys; on_change triggers re‑init)
st.sidebar.selectbox(
    "A1 Sensor LoA",
    ["Human", "Hybrid", "Automated"],
    key="role_A1",
    index=["Human", "Hybrid", "Automated"].index(st.session_state.roles.get("A1", "Automated")),
    on_change=mark_need_reinit,
    help="Select the level of automation for the Sensor. "
         "Human: slower & lower reliability, Automated: faster & higher reliability. "
         "The final values used come from the per‑role × LoA defaults below."
)
st.sidebar.selectbox(
    "A2 Analyst LoA",
    ["Human", "Hybrid", "Automated"],
    key="role_A2",
    index=["Human", "Hybrid", "Automated"].index(st.session_state.roles.get("A2", "Hybrid")),
    on_change=mark_need_reinit,
    help="Select the level of automation for the Analyst. "
         "This affects how recommendations are generated (reliability/latency)."
)
st.sidebar.selectbox(
    "A3 Comms LoA",
    ["Human", "Hybrid", "Automated"],
    key="role_A3",
    index=["Human", "Hybrid", "Automated"].index(st.session_state.roles.get("A4", "Human")),
    on_change=mark_need_reinit,
    help="Select the level of automation for the Communications agent. "
         "This affects delivery reliability and latency along the pipeline."
)
st.sidebar.selectbox(
    "A4 Decider LoA",
    ["Human", "Hybrid", "Automated"],
    key="role_A4",
    index=["Human", "Hybrid", "Automated"].index(st.session_state.roles.get("A3", "Human")),
    on_change=mark_need_reinit,
    help="Select the level of automation for the Decider. "
         "This drives the Decider's action reliability (collect/act)."
)

roles = {
    'A1': st.session_state.role_A1,
    'A2': st.session_state.role_A2,
    'A3': st.session_state.role_A3,
    'A4': st.session_state.role_A4,
}

# Initial trust sliders
p['trust_A1_A2'] = st.sidebar.slider(
    "Initial trust: Analyst (A2) in Sensor (A1)",
    -1.0, 1.0, p['trust_A1_A2'], 0.1,
    help="Initial trust on link A1→A2 (−1..1). Higher trust increases effective reliability and lowers effective latency."
)
p['trust_A2_A3'] = st.sidebar.slider(
    "Initial trust: Comms (A3) in Analyst (A2)",
    -1.0, 1.0, p['trust_A2_A3'], 0.1,
    help="Initial trust on link A2→A3 (−1..1). Higher trust increases delivery success and reduces communication delay."
)
p['trust_A3_A4'] = st.sidebar.slider(
    "Initial trust: Decider (A4) in Comms (A3)",
    -1.0, 1.0, p['trust_A3_A4'], 0.1,
    help="Initial trust on link A3→A4 (−1..1). Higher trust increases action reliability and supports faster OODA cycles."
)


# ======================================================================================
# Sidebar: Per‑role × LoA editable table
# ======================================================================================
defaults_per_role = st.session_state.defaults_per_role

with st.sidebar.expander("Per‑role × LoA defaults (editable table)"):
    st.caption(
        "Edit **reliability** (0..1) and **latency** (0..12 steps) for each role × LoA. "
        "These values are applied when you select a LoA above. Click **Save** to store and apply."
    )


    def defaults_dict_to_df(d: dict) -> pd.DataFrame:
        rows = {}
        for role, loa_dict in d.items():
            rows[role] = {
                "Human_rel": loa_dict["Human"]["reliability"],
                "Human_lat": loa_dict["Human"]["latency"],
                "Hybrid_rel": loa_dict["Hybrid"]["reliability"],
                "Hybrid_lat": loa_dict["Hybrid"]["latency"],
                "Automated_rel": loa_dict["Automated"]["reliability"],
                "Automated_lat": loa_dict["Automated"]["latency"],
            }
        return pd.DataFrame.from_dict(rows, orient="index")


    df_defaults = defaults_dict_to_df(defaults_per_role)

    editor = get_data_editor()
    edited_df = editor(
        df_defaults,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Human_rel":      st.column_config.NumberColumn("Human rel",      min_value=0.0, max_value=1.0, step=0.01, format="%.2f", help="Reliability for Human LoA"),
            "Human_lat":      st.column_config.NumberColumn("Human lat",      min_value=0,   max_value=12,  step=1,                   help="Latency (steps) for Human LoA"),
            "Hybrid_rel":     st.column_config.NumberColumn("Hybrid rel",     min_value=0.0, max_value=1.0, step=0.01, format="%.2f", help="Reliability for Hybrid LoA"),
            "Hybrid_lat":     st.column_config.NumberColumn("Hybrid lat",     min_value=0,   max_value=12,  step=1,                   help="Latency (steps) for Hybrid LoA"),
            "Automated_rel":  st.column_config.NumberColumn("Automated rel",  min_value=0.0, max_value=1.0, step=0.01, format="%.2f", help="Reliability for Automated LoA"),
            "Automated_lat":  st.column_config.NumberColumn("Automated lat",  min_value=0,   max_value=12,  step=1,                   help="Latency (steps) for Automated LoA"),
        }
    )

    c1, c3 = st.columns([1, 1])
    with c1:
        if st.button("Save", type="primary", use_container_width=True, help="Save changes to per‑role × LoA defaults and apply them immediately."):
            st.session_state.defaults_per_role = _df_to_defaults(edited_df)
            # Apply to params and mark need_reinit so figures reflect it immediately
            apply_role_loa_to_params(p, st.session_state.defaults_per_role, roles)
            st.session_state.need_reinit = True
            st.success("Per‑role × LoA defaults saved and applied.")
            safe_rerun()

    with c3:
        # Export defaults as JSON (optional)
        json_bytes = io.BytesIO(json.dumps(st.session_state.defaults_per_role, indent=2).encode("utf-8"))
        st.download_button(
            "Export JSON",
            data=json_bytes,
            file_name="role_loa_defaults.json",
            mime="application/json",
            use_container_width=True,
            help="Download the current per‑role × LoA defaults as a JSON file."
        )

# =================================================================
# Sidebar: Baseline mode + Simulation settings
# =================================================================

st.sidebar.subheader("Environment & Simulation")


if "baseline_choice" not in st.session_state:
    st.session_state.baseline_choice = "Synthetic (event realistic)"


baseline_choice = st.sidebar.selectbox(
    "Baseline source",
    ["Synthetic (sinusoidal)", "Synthetic (event realistic)",
     "File 1", "File 2", "File 3", "File 4"],
    key="baseline_choice",
    on_change=mark_need_reinit
)


# Store baseline_mode on Params
if baseline_choice == "Synthetic (sinusoidal)":
    p["baseline_mode"] = "sinusoidal"
elif baseline_choice == "Synthetic (event realistic)":
    p["baseline_mode"] = "realistic"

# Seed and steps
p['seed'] = st.sidebar.number_input(
    "Random seed",
    min_value=0, max_value=10_000_000, value=p['seed'], step=1,
    help="Controls the pseudo‑random sequence for the simulation."
)
p['steps'] = st.sidebar.slider(
    "Simulation steps", 10, 480, p['steps'], 1,
    help="Number of discrete time steps to simulate."
)

# Look‑ahead (L) bounded by steps‑1
max_L = max(0, int(p['steps']) - 1)
p['sensor_lookahead_steps'] = st.sidebar.slider(
    "Sensor look‑ahead (steps ahead)", 0, max_L,
    p.get('sensor_lookahead_steps', 4), 1,
    help="A1 Sensor observes/predicts at time t+L instead of t. "
         "This is a short‑horizon forecast proxy used by the Analyst."
)


# # A1 look-ahead
# max_L = max(0, p["steps"] - 1)
# p["sensor_lookahead_steps"] = st.sidebar.slider(
#     "Sensor look-ahead L", 0, max_L, p["sensor_lookahead_steps"]
# )

p['threshold'] = st.sidebar.slider(
    "Density risk threshold (p/m²)", 0.0, 100.0, p['threshold'], 0.5,
    help="Global risk threshold. For crowd safety, typical triggers are calibrated to Level of Service bands."
)




# --- Action effect parameters (A4) ---
p['action_effect_w_current'] = st.sidebar.slider(
    "Action effect weight – current level",
    0.0, 1.0,
    p.get('action_effect_w_current', 0.05),
    0.01,
    help="Controls how much the action reduces density based on the CURRENT ground-truth density."
)

p['action_effect_w_incoming'] = st.sidebar.slider(
    "Action effect weight – incoming change",
    0.0, 1.0,
    p.get('action_effect_w_incoming', 0.9),
    0.01,
    help="Controls how much the action reduces density based on the NEXT-step incoming trend (baseline[t+1] - baseline[t])."
)


# Advanced environment settings
with st.sidebar.expander("Advanced Environment Settings", expanded=False):
    p['amplitude'] = st.slider(
        "Sinus amplitude", 0.2, 4.0, p['amplitude'], 0.05,
        help="Amplitude of the synthetic baseline sinusoid (only used in Synthetic baseline mode)."
    )
    p['noise_sigma'] = st.slider(
        "Noise σ", 0.05, 4.0, p['noise_sigma'], 0.05,
        help="Standard deviation of Gaussian noise added to the baseline (Synthetic mode)."
    )
    p['base_level'] = st.slider(
        "Sinus base level", 1.0, 6.0, p['base_level'], 0.05,
        help="Vertical offset (mean level) of the synthetic baseline."
    )
    p['damp_steps'] = st.slider(
        "Dampening window (steps)", 0, 6, p['damp_steps'], 1,
        help="Number of steps over which dampening/containment blends future densities after an action."
    )
    tick_ms = st.slider(
        "Tick speed (ms while running)", 50, 1500,
        st.session_state.get('tick_ms', 300), 50,
        help="Visual refresh interval during Run mode."
    )
    st.session_state['tick_ms'] = tick_ms


# Load baseline files (if chosen)
baseline_df = None
if baseline_choice.startswith("File"):
    file_map = {
        "File 1": r"C:\Users\tldert\PycharmProjects\CrisisManagement\data\SAIL2025\merged_daily_12minAHEAD\chronos_forecast_results_GASA-01-B_315\chronos_forecast_results_GASA-01-B_315__2025-08-21_ahead3.csv",
        "File 2": r"C:\Users\tldert\PycharmProjects\CrisisManagement\data\SAIL2025\merged_daily_12minAHEAD\chronos_forecast_results_GASA-01-B_315\chronos_forecast_results_GASA-01-B_315__2025-08-22_ahead3.csv",
        "File 3": r"C:\Users\tldert\PycharmProjects\CrisisManagement\data\SAIL2025\merged_daily_12minAHEAD\chronos_forecast_results_GASA-01-B_315\chronos_forecast_results_GASA-01-B_315__2025-08-23_ahead3.csv",
        "File 4": r"C:\Users\tldert\PycharmProjects\CrisisManagement\data\SAIL2025\merged_daily_12minAHEAD\chronos_forecast_results_GASA-01-B_315\chronos_forecast_results_GASA-01-B_315__2025-08-24_ahead3.csv"
    }
    path = file_map[baseline_choice]
    if os.path.exists(path):
        baseline_df = pd.read_csv(path)
        st.sidebar.success(f"Loaded {path}")
    else:
        st.sidebar.error(f"File not found: {path}")

st.session_state.baseline_df = baseline_df


# =================================================================
# Apply LoA → Params and handle auto-reinit
# =================================================================
apply_role_loa_to_params(p, st.session_state.defaults_per_role, roles)
st.session_state.params = p

# Simple init: only initialize once
if "state" not in st.session_state:
    init_or_reinit_state(baseline_df=st.session_state.baseline_df)


# =================================================================
# Controls: run / pause / step / fast-forward + reset
# =================================================================
col_btn1, col_btn2, col_btn3, col_btn5, col_btn_reset = st.columns(5)

# Run
if col_btn1.button("▶ Run"):
    st.session_state.running = True

# Pause
if col_btn2.button("⏸ Pause"):
    st.session_state.running = False

# Step
if col_btn3.button("⏭ Step") and not st.session_state.get("running", False):
    st.session_state.state = step_once(st.session_state.state, Params(**st.session_state.params))

# Skip to end
if col_btn5.button("⏭ Skip to End"):
    st.session_state.running = False
    params_obj = Params(**st.session_state.params)
    while st.session_state.state.t < len(st.session_state.state.baseline) - 1:
        st.session_state.state = step_once(st.session_state.state, params_obj)
    safe_rerun()

# NEW: Reset simulation
if col_btn_reset.button("Reset Simulation"):
    apply_role_loa_to_params(st.session_state.params, st.session_state.defaults_per_role, st.session_state.roles)
    init_or_reinit_state(baseline_df=st.session_state.baseline_df)
    st.session_state.running = False
    st.success("Simulation reset with current parameters.")
    safe_rerun()

# =================================================================
# Tabs & Figures
# =================================================================
tab_main, tab_analysis = st.tabs(["Operations", "Trust Analysis"])

with tab_main:
    L = p["sensor_lookahead_steps"]
    mps = p["minutes_per_step"]
    st.caption(f"Sensor lookahead L={L} (~{L*mps} min)")

    col_net, col_ts = st.columns(2)
    with col_net:
        st.plotly_chart(sna_figure(st.session_state.state, Params(**p)), use_container_width=True)
    with col_ts:
        st.plotly_chart(timeseries_figure(st.session_state.state, Params(**p)),
                         use_container_width=True)

with tab_analysis:
    st.plotly_chart(trust_timeseries_figure(st.session_state.state, Params(**p)),
                     use_container_width=True)


# =================================================================
# Metrics
# =================================================================
state = st.session_state.state
params_obj = Params(**st.session_state.params)
threshold_val = p["threshold"]

X = state.T_MINUTES if hasattr(state, "T_MINUTES") else np.arange(0, len(state.baseline)*params_obj.minutes_per_step, params_obj.minutes_per_step)
t_now = min(state.t - 1, len(X) - 1)
violations = np.sum(state.density[:t_now+1] > threshold_val)
total_actions = int(np.sum(state.decider_action))

m1, m2, m3 = st.columns(3)
m1.metric("Current minute", f"{X[t_now]} min")
m2.metric("Total actions", total_actions)
m3.metric("Violations", violations)


# =================================================================
# Run loop
# =================================================================
tick_ms = st.session_state.get("tick_ms", 300)
max_t = len(state.baseline) - 1

if st.session_state.get("running", False):
    if state.t < max_t:
        st.session_state.state = step_once(st.session_state.state, Params(**p))
        time.sleep(tick_ms/1000)
        safe_rerun()
    else:
        st.session_state.running = False
