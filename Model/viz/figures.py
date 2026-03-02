
# viz/figures.py
"""
Plotly figure builders for the H‑AI crowd ABM.
Each function returns a plotly.graph_objects.Figure for Streamlit to render.

This version visualizes A1 look-ahead forecasts:
- The sensor at time t produces a prediction for time t+L (L = sensor_lookahead_steps).
- We plot BOTH:
    1) the prediction issued at time t (unshifted),
    2) the same prediction placed at its target time (shifted by +L steps).
- Optional "link" segments can be drawn from (t) -> (t+L) to illustrate the forecast mapping.
"""

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

# Only import the types/classes (no global constants)
from Model.model.core import SimState, Params

# ---- Small helpers to read dynamic timeline & threshold ----
def _get_time_axis(state: SimState, params: Params) -> np.ndarray:
    """Return the minutes timeline for plotting."""
    if hasattr(state, "T_MINUTES") and isinstance(state.T_MINUTES, np.ndarray):
        return state.T_MINUTES
    # Fallback: build from series length and params.minutes_per_step (default 3)
    mps = getattr(params, "minutes_per_step", 3)
    steps = len(getattr(state, "baseline", [])) or len(getattr(state, "density", []))
    return np.arange(0, steps * mps, mps)

def _get_threshold(params: Params) -> float:
    """Return dynamic threshold stored in params; fallback to 4.0."""
    return float(getattr(params, "threshold", 4.0))


# ======================================================================
# SNA / Network figure
# ======================================================================
def sna_figure(state: SimState, params: Params) -> go.Figure:
    """
    Build a static network (SNA) figure representing the 4-agent pipeline at the current tick.
    Now annotates A1's look-ahead (sensor_lookahead_steps).
    """
    X = _get_time_axis(state, params)
    TH = _get_threshold(params)
    t = max(0, min(state.t - 1, len(X) - 1))

    L = int(getattr(params, "sensor_lookahead_steps", 4))  # <— LOOK-AHEAD

    # Fixed positions
    pos = {
        'Env': (0.50, 0.75),
        'Sensor': (0.15, 0.35),
        'Analyst': (0.40, 0.35),
        'Comms': (0.65, 0.35),
        'Decider': (0.85, 0.35),
    }
    edges = [
        ('Env', 'Sensor'),
        ('Sensor', 'Analyst'),
        ('Analyst', 'Comms'),
        ('Comms', 'Decider'),
        ('Decider', 'Env')
    ]

    # Node colors (density v. threshold)
    env_color = (
        '#2ca02c' if state.density[t] < 3.5
        else ('#ff7f0e' if state.density[t] < TH else '#d62728')
    )
    sensor_color = '#1f77b4' if state.sensor_acc[t] else '#8fbce6'
    analyst_color = '#ff7f0e' if state.analyst_recommend[t] else '#dddddd'
    comms_color = '#9467bd' if state.comms_success[t] else '#dddddd'
    decider_color = (
        '#9ACD32' if (state.decider_action[t] and state.action_success[t])
        else ('#2ca02c' if state.decider_action[t] else '#dddddd')
    )

    nodes = [
        ('Env', env_color), ('Sensor', sensor_color), ('Analyst', analyst_color),
        ('Comms', comms_color), ('Decider', decider_color)
    ]

    node_x, node_y, node_text, node_color = [], [], [], []
    for name, color in nodes:
        x, y = pos[name]
        node_x.append(x); node_y.append(y)
        node_text.append(name); node_color.append(color)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=node_text, textposition='bottom center',
        textfont=dict(size=20),
        marker=dict(size=30, color=node_color, line=dict(color='black', width=1)),
        hoverinfo='text',
        name="Agents"
    )

    # Base edges (light gray)
    edge_traces = []
    for u, v in edges:
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_traces.append(
            go.Scatter(
                x=[x0, x1], y=[y0, y1], mode='lines',
                line=dict(width=2, color='lightgray'),
                hoverinfo='none', showlegend=False
            )
        )

    # Highlights for current tick
    highlight_traces = []
    # Sensor -> Analyst highlight when sensor info is available (A1 latency)
    if t - params.A1_latency >= 0:
        x0, y0 = pos['Sensor']; x1, y1 = pos['Analyst']
        highlight_traces.append(
            go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                       line=dict(width=4, color='blue'),
                       hoverinfo='none', showlegend=False)
        )
    # Analyst -> Comms highlight when recommendation issued at t
    if state.analyst_recommend[t]:
        x0, y0 = pos['Analyst']; x1, y1 = pos['Comms']
        highlight_traces.append(
            go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                       line=dict(width=4, color='orange'),
                       hoverinfo='none', showlegend=False)
        )
    # Comms -> Decider highlight when delivery arrives at t
    if state.comms_success[t]:
        x0, y0 = pos['Comms']; x1, y1 = pos['Decider']
        highlight_traces.append(
            go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                       line=dict(width=4, color='purple'),
                       hoverinfo='none', showlegend=False)
        )
    # Decider -> Env highlight when action occurs at t
    if state.decider_action[t]:
        x0, y0 = pos['Decider']; x1, y1 = pos['Env']
        highlight_traces.append(
            go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                       line=dict(width=4, color='green'),
                       hoverinfo='none', showlegend=False)
        )

    # Trust labels above edges (current scalar trust values)
    def edge_label(u, v, text, color, size=20, dy=0.04):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0 + dy
        return dict(
            x=xm, y=ym, xref='x', yref='y',
            text=text, showarrow=False,
            font=dict(size=size, color=color),
            align='center',
            bgcolor='rgba(255,255,255,0.6)',
            bordercolor=color, borderwidth=0.5
        )

    annotations = [
        # Show latency/reliability and LOOK-AHEAD on the A1 edge:
        edge_label('Sensor',  'Analyst',
                   f"(lat={params.A1_latency}, rel={params.A1_reliability:.2f}, ahead={L})",
                   'blue'),
        edge_label('Analyst', 'Comms',
                   f"(lat={params.A2_latency}, rel={params.A2_reliability:.2f})", 'orange'),
        edge_label('Comms',   'Decider',
                   f"(lat={params.A3_latency}, rel={params.A3_reliability:.2f})", 'purple'),
        edge_label('Sensor',  'Analyst', f"trust={state.trust_A1_A2:.2f}", 'blue', dy=0.10),
        edge_label('Analyst', 'Comms',   f"trust={state.trust_A2_A3:.2f}", 'orange', dy=0.10),
        edge_label('Comms',   'Decider', f"trust={state.trust_A3_A4:.2f}", 'purple', dy=0.10),

        # --- NEW: A4 (Decider -> Env) annotation ---
        edge_label('Decider', 'Env',
                   f"(lat={params.A4_latency}, rel={params.A4_reliability:.2f})", 'green'),
        #edge_label('Comms',   'Decider', f"Effect(current,income)=-{params.action_effect_w_current}, {params.action_effect_w_incoming}:.2f}", 'green', dy=0.10),

    ]

    env_label = 'GREEN' if state.density[t] < 3.5 else ('ORANGE' if state.density[t] < TH else 'RED')
    status_text = {'GREEN': 'safe (green)', 'ORANGE': 'caution needed (orange)', 'RED': 'risky (red)'}[env_label]
    t_label = X[t] if len(X) else 0

    fig = go.Figure(data=edge_traces + highlight_traces + [node_trace])
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        title=f"SNA — t={t_label} min | Env= {status_text}",
        showlegend=False, height=420, annotations=annotations
    )
    return fig


# ======================================================================
# Trust time series figure (clean, only trust, bold + clear)
# ======================================================================
def trust_timeseries_figure(state: SimState, params: Params) -> go.Figure:
    """
    Clean trust-only time series.
    Shows:
      - Trust A1→A2
      - Trust A2→A3
      - Trust A3→A4
    All as thick, highly visible lines.
    """
    X = _get_time_axis(state, params)
    t = max(0, min(state.t - 1, len(X) - 1))

    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Unified style for bold trust curves
    TRUST_WIDTH = 3.5
    TRUST_OPACITY = 0.95

    # A1 -> A2
    fig.add_trace(go.Scatter(
        x=X[:t+1], y=state.trust_A1_A2_hist[:t+1],
        name="Trust A1→A2",
        mode="lines",
        line=dict(color="blue", width=8),
        opacity=TRUST_OPACITY,
        hovertemplate="Trust A1→A2: %{y:.2f} at %{x} min<extra></extra>",
    ))

    # A2 -> A3
    fig.add_trace(go.Scatter(
        x=X[:t+1], y=state.trust_A2_A3_hist[:t+1],
        name="Trust A2→A3",
        mode="lines",
        line=dict(color="orange", width=5),
        opacity=TRUST_OPACITY,
        hovertemplate="Trust A2→A3: %{y:.2f} at %{x} min<extra></extra>",
    ))

    # A3 -> A4
    fig.add_trace(go.Scatter(
        x=X[:t+1], y=state.trust_A3_A4_hist[:t+1],
        name="Trust A3→A4",
        mode="lines",
        line=dict(color="purple", width=2),
        opacity=TRUST_OPACITY,
        hovertemplate="Trust A3→A4: %{y:.2f} at %{x} min<extra></extra>",
    ))

    # ----- Axes & layout -----
    x_max = float(X[-1]) if len(X) else 0.0
    x_dtick = x_max / 10.0 if x_max > 0 else 1.0

    fig.update_xaxes(
        title_text="Time (min)",
        range=[0, x_max],
        tickmode="linear",
        dtick=x_dtick,
        showline=True,
        linecolor="black",
        showgrid=True
    )

    fig.update_yaxes(
        title_text="Trust (−1..1)",
        range=[-1.05, 1.05],
        tickmode="linear",
        dtick=0.2,
        showline=True,
        linecolor="black",
    )

    t_label = X[t] if len(X) else 0

    # LEGEND in top-right, vertical
    fig.update_layout(
        margin=dict(l=10, r=40, t=50, b=10),
        title=f"Trust Timeline — t={X[t] if len(X) else 0} min",
        hovermode="x unified",
        height=420,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1.15   # slightly outside plot
        )
    )


    return fig


# ======================================================================
# Density / timeline figure with look-ahead forecast visualization
# ======================================================================
def timeseries_figure(state: SimState, params: Params | None = None) -> go.Figure:
    """
    Build a single-axis timeline showing baseline, density, threshold, violations,
    actions, AND the A1 look-ahead predictions.

    If `params` is provided, reads `sensor_lookahead_steps` from it; else defaults to 4.
    Two A1 lines are plotted:
      - "Sensor (issued @ t → predicts t+L)": sensor_meas plotted at current time t (unshifted).
      - "Forecast placed @ target (t+L)": the same values shifted forward by L steps (clipped to range).
    Optional "link" segments illustrate the mapping from (t) → (t+L).
    """
    # Threshold
    if params is not None:
        TH = _get_threshold(params)
        X = _get_time_axis(state, params)
        L = int(getattr(params, "sensor_lookahead_steps", 4))
    else:
        TH = float(getattr(state, "threshold", 4.0))
        steps = len(state.baseline)
        X = np.arange(0, steps * 3, 3)  # fallback assumes 3-min steps
        L = 4

    t = max(0, min(state.t - 1, len(X) - 1))
    n = len(X)

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    # ---------------- Base series ----------------
    fig.add_trace(go.Scatter(
        x=X, y=state.baseline, name="Baseline",
        mode="lines",
        line=dict(color="gray", width=1.5),
        hovertemplate="Baseline %{y:.2f} p/m² at %{x} min<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=X[:t+1], y=state.density[:t+1], name="Density",
        mode="lines+markers",
        line=dict(color="steelblue", width=1.7),
        marker=dict(size=6, color="steelblue"),
        hovertemplate="Density %{y:.2f} p/m² at %{x} min<extra></extra>",
    ), secondary_y=False)

    # ---------------- A1 look-ahead (two views) ----------------
    # 1) Issued @ t (unshifted): this shows when the prediction was made
    sensor_y = state.sensor_meas[:t+1]
    sensor_x = X[:t+1]
    valid_now = ~np.isnan(sensor_y)
    # if np.any(valid_now):
    #     fig.add_trace(go.Scatter(
    #         x=sensor_x[valid_now], y=sensor_y[valid_now],
    #         name=f"Sensor (issued @ t → predicts t+{L})",
    #         mode="lines",
    #         line=dict(color="darkorange", width=1.8),
    #         opacity=0.65,
    #         hovertemplate="Prediction (issued @ %{x} min): %{y:.2f} p/m²<extra></extra>",
    #     ), secondary_y=False)

    # 2) Placed @ target time (shifted by +L): this shows where that prediction applies
    if L > 0:
        idx_src = np.arange(0, min(t + 1, n))                  # times the predictions were made
        idx_tgt = idx_src + L                                  # target times
        mask_in = idx_tgt < n                                  # keep only those that fit on the axis
        if np.any(mask_in):
            fx = X[idx_tgt[mask_in]]
            fy = state.sensor_meas[idx_src[mask_in]]
            valid_forecast = ~np.isnan(fy)
            if np.any(valid_forecast):
                fig.add_trace(go.Scatter(
                    x=fx[valid_forecast], y=fy[valid_forecast],
                    name=f"Forecast placed @ target (t+{L})",
                    mode="lines",
                    line=dict(color="orange", width=1.5),
                    opacity=0.9,
                    hovertemplate="Forecast for %{x} min: %{y:.2f} p/m²<extra></extra>",
                ), secondary_y=False)

                # OPTIONAL link segments (from issue time to target time).
                # Comment out the block below if you prefer not to draw links.
                link_opacity = 0.15
                max_links = 40  # limit for performance
                count = 0
                for i_src, i_tgt in zip(idx_src[mask_in], idx_tgt[mask_in]):
                    if count >= max_links:
                        break
                    y_val = state.sensor_meas[i_src]
                    if np.isnan(y_val):
                        continue
                    fig.add_shape(
                        type="line",
                        x0=X[i_src], y0=y_val,
                        x1=X[i_tgt], y1=y_val,
                        line=dict(color="orange", width=1, dash="dot"),
                        opacity=link_opacity,
                        layer="below"
                    )
                    count += 1

    # ---------------- Threshold & violations ----------------
    fig.add_hline(
        y=TH, line_dash="dot", line_color="red",
        annotation_text=f"Threshold {TH:.2f} p/m²",
        annotation_position="top left"
    )

    viol_mask = state.density[:t+1] > TH
    if np.any(viol_mask):
        fig.add_trace(go.Scatter(
            x=X[:t+1][viol_mask], y=state.density[:t+1][viol_mask],
            name="Violations", mode="markers",
            marker=dict(color="red", size=8, opacity=0.6),
            hovertemplate="Violation: %{y:.2f} p/m² at %{x} min<extra></extra>",
        ), secondary_y=False)

    # ---------------- Actions & markers ----------------
    y_action = state.decider_action[:t+1].astype(int)
    if len(y_action) > 0:
        fig.add_trace(go.Scatter(
            x=X[:t+1], y=y_action, name="Action (0/1)",
            mode="lines", line=dict(color="green", width=1.5),
            line_shape="hv", opacity=0.6,
            hovertemplate="Action=%{y} at %{x} min<extra></extra>",
        ), secondary_y=False)

    x_rec = X[:t+1][state.analyst_recommend[:t+1]]
    if len(x_rec) > 0:
        fig.add_trace(go.Scatter(
            x=x_rec, y=np.ones_like(x_rec) * (TH * 0.05 + 0.05),  # small offset line
            name="Recommend", mode="markers",
            marker=dict(symbol="triangle-down", color="orange", size=9, opacity=0.6),
            hovertemplate="Recommend at %{x} min<extra></extra>",
        ), secondary_y=False)

    x_del = X[:t+1][state.comms_success[:t+1]]
    if len(x_del) > 0:
        fig.add_trace(go.Scatter(
            x=x_del, y=np.ones_like(x_del) * (TH * 0.07 + 0.08),
            name="Delivered", mode="markers",
            marker=dict(symbol="star", color="purple", size=9, opacity=0.6),
            hovertemplate="Delivered at %{x} min<extra></extra>",
        ), secondary_y=False)

    # ---------------- Axes & layout ----------------
    y_max = max(6.0, TH * 1.6)
    y_dtick = max(0.1, TH / 10.0)
    x_max = float(X[-1]) if len(X) else 0.0
    x_dtick = x_max / 10.0 if x_max > 0 else 1.0

    # ----- Globale fontinstellingen -----
    BASE = 18  # algemene basisgrootte
    AXIS_TITLE = 20  # as-titels
    TICK = 15  # tick labels
    LEGEND = 18  # legenda
    TITLE = 20  # figuurtitel
    HOVER = 18  # hover labels
    ANNOT = 18  # annotaties (zoals bij add_hline)


    fig.update_layout(
        font=dict(size=BASE),  # basis voor alle tekst die geen eigen font heeft
        title=dict(
            text=f"Crowd density + Forecast timeline",
            font=dict(size=TITLE)
        ),
        legend=dict(
            font=dict(size=LEGEND),
            orientation="v", yanchor="bottom", y=1.02, xanchor="right", x=1
        ),
        hoverlabel=dict(
            font=dict(size=HOVER)
        ),
    )

    # As-titels en ticks
    fig.update_xaxes(
        title_text="Time (min)",
        title_font=dict(size=AXIS_TITLE),
        tickfont=dict(size=TICK)
    )
    fig.update_yaxes(
        title_text="Persons per m²",
        title_font=dict(size=AXIS_TITLE),
        tickfont=dict(size=TICK)
    )

    # Annotatie die uit add_hline komt: lettertype groter maken
    # (update_annotations werkt op alle bestaande annotation-objects)
    fig.update_annotations(font=dict(size=ANNOT))


    # fig.update_yaxes(
    #     title_text="Persons per m²",
    #     range=[0, y_max],
    #     tickmode="linear",
    #     dtick=y_dtick
    # )
    # fig.update_xaxes(
    #     title_text="Time (min)",
    #     range=[0, x_max],
    #     tickmode="linear",
    #     dtick=x_dtick,
    #     showline=True, linecolor="black",
    #     showgrid=True
    # )
    #
    # t_label = X[t] if len(X) else 0
    # fig.update_layout(
    #     margin=dict(l=10, r=50, t=50, b=10),
    #     title=f"Crowd density + Forecast timeline — t={t_label} min (look-ahead={L})",
    #     legend=dict(orientation="v", yanchor="bottom", y=1.02, xanchor="right", x=1),
    #     hovermode="x unified",
    #     height=480,
    # )
    return fig