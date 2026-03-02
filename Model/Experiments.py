
# experiment.py
"""
Scenario-based experiments for the H-AI Crowd ABM.

Runs predefined LoA + foresight + trust configurations and produces:
1) A single density time series figure (each scenario in a different color)
2) A single trust time series figure (one trust line per scenario)

Intended for analysis, papers, and controlled comparisons (no Streamlit).
"""

from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from model.core import Params, init_state, step_once


# ============================================================
# Scenario definitions
# ============================================================

SCENARIOS = [
    dict(
        name="Baseline Coordination",
        loa=dict(A1="Human", A2="Human", A3="Human", A4="Human"),
        lookahead_min=0,
        init_trust=+0.5,
    ),
    dict(
        name="Reactive (High trust)",
        loa=dict(A1="Automated", A2="Hybrid", A3="Hybrid", A4="Human"),
        lookahead_min=0,
        init_trust=+0.5,
    ),
    dict(
        name="Hybrid Foresight (High trust)",
        loa=dict(A1="Automated", A2="Hybrid", A3="Hybrid", A4="Human"),
        lookahead_min=12,
        init_trust=+0.5,
    ),
    dict(
        name="Hybrid Foresight (Low trust)",
        loa=dict(A1="Automated", A2="Hybrid", A3="Hybrid", A4="Human"),
        lookahead_min=12,
        init_trust=-0.5,
    ),
    dict(
        name="Automation-driven Foresight",
        loa=dict(A1="Automated", A2="Automated", A3="Automated", A4="Hybrid"),
        lookahead_min=12,
        init_trust=+0.5,
    ),
]


# ============================================================
# LoA defaults (same logic as app)
# ============================================================

DEFAULTS_PER_ROLE = {
    "Sensor": {
        "Human":     {"reliability": 0.40, "latency": 5},
        "Hybrid":    {"reliability": 0.70, "latency": 3},
        "Automated": {"reliability": 0.95, "latency": 0},
    },
    "Analyst": {
        "Human":     {"reliability": 0.80, "latency": 3},
        "Hybrid":    {"reliability": 0.90, "latency": 2},
        "Automated": {"reliability": 0.70, "latency": 0},
    },
    "Comms": {
        "Human":     {"reliability": 0.60, "latency": 2},
        "Hybrid":    {"reliability": 0.75, "latency": 1},
        "Automated": {"reliability": 0.90, "latency": 0},
    },
    "Decider": {
        "Human":     {"reliability": 0.90, "latency": 4},
        "Hybrid":    {"reliability": 0.85, "latency": 3},
        "Automated": {"reliability": 0.70, "latency": 0},
    },
}


def apply_loa(params: dict, loa: dict):
    """Apply LoA regime to Params dict."""
    params["A1_reliability"] = DEFAULTS_PER_ROLE["Sensor"][loa["A1"]]["reliability"]
    params["A1_latency"]     = DEFAULTS_PER_ROLE["Sensor"][loa["A1"]]["latency"]

    params["A2_reliability"] = DEFAULTS_PER_ROLE["Analyst"][loa["A2"]]["reliability"]
    params["A2_latency"]     = DEFAULTS_PER_ROLE["Analyst"][loa["A2"]]["latency"]

    params["A3_reliability"] = DEFAULTS_PER_ROLE["Comms"][loa["A3"]]["reliability"]
    params["A3_latency"]     = DEFAULTS_PER_ROLE["Comms"][loa["A3"]]["latency"]

    a4_rel = DEFAULTS_PER_ROLE["Decider"][loa["A4"]]["reliability"]
    params["A4_reliability"] = a4_rel


# ============================================================
# Run one scenario
# ============================================================
N_RUNS = 1000              # Monte-Carlo runs per scenario
CONF_INT = (10, 90)         # percentile bands
def run_scenario_mc(scn: dict, base_params: Params, n_runs: int):
    states = []

    for i in range(n_runs):
        p = asdict(base_params)

        # --- seed changes per replication ---
        p["seed"] = base_params.seed + i

        # Apply LoA
        apply_loa(p, scn["loa"])

        # # Look-ahead (minutes → steps)
        # p["sensor_lookahead_steps"] = int(
        #     scn["lookahead_min"] / p["minutes_per_step"]
        # )
        p["sensor_lookahead_steps"]= scn["lookahead_min"]

        # Initial trust (uniform)
        p["trust_A1_A2"] = scn["init_trust"]
        p["trust_A2_A3"] = scn["init_trust"]
        p["trust_A3_A4"] = scn["init_trust"]

        params_obj = Params(**p)
        state = init_state(params_obj)

        while state.t < state.T - 1:
            state = step_once(state, params_obj)

        states.append(state)

    return states

def run_scenario(scn: dict, base_params: Params):
    p = asdict(base_params)

    # Apply LoA
    apply_loa(p, scn["loa"])

    # Look-ahead (minutes → steps)
    p["sensor_lookahead_steps"] = int(
        scn["lookahead_min"] / p["minutes_per_step"]
    )

    # Initial trust (all links equal)
    p["trust_A1_A2"] = scn["init_trust"]
    p["trust_A2_A3"] = scn["init_trust"]
    p["trust_A3_A4"] = scn["init_trust"]

    params_obj = Params(**p)
    state = init_state(params_obj)

    while state.t < state.T - 1:
        state = step_once(state, params_obj)

    return state


# ============================================================
# Figures
# ============================================================
import plotly.graph_objects as go
import numpy as np

# --- Palette per scenario (customize as you like) ---
SCENARIO_COLORS = {
    "Baseline Coordination":           "#4C78A8",  # blue
    "Reactive (High trust)":           "#F58518",  # orange
    "Hybrid Foresight (High trust)":   "#54A24B",  # green
    "Hybrid Foresight (Low trust)":    "#E45756",  # red
    "Automation-driven Foresight":     "#B279A2",  # purple
}

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(r,g,b,alpha)'."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def density_comparison_figure_mc(
    results_mc: dict,
    params: Params,
    conf_int=(10, 90),
    band_style: str = "fill",   # "fill" (translucent bands) or "lines" (dotted quantiles)
    band_alpha: float = 0.15,   # transparency for fill
    band_line_dash: str = "dot" # dash style for "lines" mode: "dot" | "dash" | "dashdot"
) -> go.Figure:
    """
    Plot mean density for each scenario + confidence display (fill or dotted lines),
    with shared baseline and horizontal risk threshold.
    """
    fig = go.Figure()

    # Time axis
    X = np.arange(0, params.steps * params.minutes_per_step, params.minutes_per_step)

    # --------------------------------------------------
    # Ground truth / baseline (from any replication)
    # --------------------------------------------------
    first_scenario_states = next(iter(results_mc.values()))
    first_state = first_scenario_states[0]
    fig.add_trace(go.Scatter(
        x=X, y=first_state.baseline,
        name="Ground truth (baseline)",
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        hovertemplate="Baseline %{y:.2f} p/m² at %{x} min<extra></extra>",
    ))

    # --------------------------------------------------
    # Risk threshold (horizontal)
    # --------------------------------------------------
    TH = params.threshold
    fig.add_hline(
        y=TH,
        line=dict(color="red", width=2, dash="dot"),
        annotation_text=f"Risk threshold = {TH:.2f} p/m²",
        annotation_position="top left",
    )

    # --------------------------------------------------
    # Scenarios: mean + colored CI
    # --------------------------------------------------
    for name, states in results_mc.items():
        color = SCENARIO_COLORS.get(name, "#666666")

        # Collect densities across replications
        D = np.array([s.density for s in states])
        mean = D.mean(axis=0)
        lo   = np.percentile(D, conf_int[0], axis=0)
        hi   = np.percentile(D, conf_int[1], axis=0)

        # Mean line (solid in scenario color)
        fig.add_trace(go.Scatter(
            x=X, y=mean,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=f"{name}<br>%{{y:.2f}} p/m² at %{{x}} min<extra></extra>",
        ))

        if band_style.lower() != "fill":
            # Translucent scenario-colored band
            fill_color = hex_to_rgba(color, band_alpha)

            fig.add_trace(go.Scatter(
                x=np.concatenate([X, X[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
        else:
            # Dotted quantile lines in scenario color (high transparency)
            fig.add_trace(go.Scatter(
                x=X, y=hi,
                mode="lines",
                name=f"{name} {conf_int[1]}th",
                line=dict(color=color, width=1, dash=band_line_dash),
                opacity=0.95,
                showlegend=False,  # keep legend clean; set True if you want explicit labels
                hovertemplate=f"{name} {conf_int[1]}th<br>%{{y:.2f}} p/m² at %{{x}} min<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=X, y=lo,
                mode="lines",
                name=f"{name} {conf_int[0]}th",
                line=dict(color=color, width=1, dash=band_line_dash),
                opacity=0.95,
                showlegend=False,
                hovertemplate=f"{name} {conf_int[0]}th<br>%{{y:.2f}} p/m² at %{{x}} min<extra></extra>",
            ))

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    fig.update_layout(
        #title=f"Crowd Density with {conf_int[0]}–{conf_int[1]}% Confidence  ({'Bands' if band_style=='fill' else 'Dotted quantiles'})",
        xaxis_title="Time (min)",
        yaxis_title="Persons per m²",
        hovermode="x unified",
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02
        ),
        font=dict(size=17),
        margin=dict(l=60, r=200, t=60, b=50),
    )

    return fig

def trust_comparison_figure_mc(
    results_mc: dict,
    params: Params,
    conf_int=(10, 90),
    band_style: str = "fill",   # "fill" or "lines"
    band_alpha: float = 0.15,
    band_line_dash: str = "dot"
) -> go.Figure:
    """
    Plot mean trust (averaged across links) per scenario
    with Monte-Carlo confidence intervals.
    """
    fig = go.Figure()

    # Time axis
    X = np.arange(0, params.steps * params.minutes_per_step, params.minutes_per_step)

    for name, states in results_mc.items():
        color = SCENARIO_COLORS.get(name, "#666666")

        # --------------------------------------------------
        # Collect trust per run (mean across links per step)
        # Shape: (n_runs, T)
        # --------------------------------------------------
        T = np.array([
            np.nanmean(
                np.vstack([
                    s.trust_A1_A2_hist,
                    s.trust_A2_A3_hist,
                    s.trust_A3_A4_hist,
                ]),
                axis=0
            )
            for s in states
        ])

        mean = T.mean(axis=0)
        lo   = np.percentile(T, conf_int[0], axis=0)
        hi   = np.percentile(T, conf_int[1], axis=0)

        # Mean line
        fig.add_trace(go.Scatter(
            x=X, y=mean,
            mode="lines",
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=(
                f"{name}<br>"
                "Trust %{y:.2f} at %{x} min"
                "<extra></extra>"
            ),
        ))
        if band_style.lower() != "fill":
            fill_color = hex_to_rgba(color, band_alpha)

            fig.add_trace(go.Scatter(
                x=np.concatenate([X, X[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=fill_color,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=X, y=hi,
                mode="lines",
                line=dict(color=color, width=1, dash=band_line_dash),
                opacity=0.95,
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=X, y=lo,
                mode="lines",
                line=dict(color=color, width=1, dash=band_line_dash),
                opacity=0.95,
                showlegend=False,
            ))

    fig.update_layout(
        #title=f"Trust Dynamics with {conf_int[0]}–{conf_int[1]}% Confidence",
        xaxis_title="Time (min)",
        yaxis_title="Trust (−1 … 1)",
        hovermode="x unified",
        margin=dict(l=60, r=200, t=60, b=50),
        legend=dict(
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
        font = dict(size=17)
        ),
    )

    return fig



# ============================================================
# KPI's table
# ============================================================
import numpy as np

def _first_non_nan(arr: np.ndarray):
    """Return first non-NaN value in arr; if all NaN, return np.nan."""
    idx = np.where(~np.isnan(arr))[0]
    return arr[idx[0]] if len(idx) else np.nan

def _last_non_nan(arr: np.ndarray):
    """Return last non-NaN value in arr; if all NaN, return np.nan."""
    idx = np.where(~np.isnan(arr))[0]
    return arr[idx[-1]] if len(idx) else np.nan

def _mean_across_links_at(arrA1, arrA2, arrA3):
    """Nan-safe mean across the three trust links at each time index."""
    stack = np.vstack([arrA1, arrA2, arrA3])
    return np.nanmean(stack, axis=0)

def _count_breach_events(series: np.ndarray, threshold: float) -> int:
    """
    Count distinct 'episodes' where series crosses above threshold.
    Example: [3.5, 4.2, 4.3, 3.9, 4.1] with TH=4.0 -> 2 events.
    """
    above = series > threshold
    # Detect rising edges (False -> True)
    return int(np.sum((~above[:-1]) & (above[1:])))

def _per_state_metrics(state, params) -> dict:
    """
    Compute requested metrics for a single simulation state.
    """
    TH = float(params.threshold)
    base_threshold = TH  # for clarity vs. code
    density = state.density
    actions = state.decider_action.astype(bool)
    delivered = state.comms_success.astype(bool)

    # 1) Actions taken (count of True)
    actions_count = int(np.sum(actions))

    # 2) Risk threshold breaches (by step)
    breach_steps = int(np.sum(density > TH))

    # Optional: breach episodes (contiguous groups)
    breach_events = _count_breach_events(density, TH)

    # 3) Recommendations delivered to decider but "no reason"
    #    We operationalize "no reason" as: density well below the safe-guard band
    #    used in act_policy: current_density < 0.7 * base_threshold
    #    (This is the anti-overreaction guard you coded.)
    no_need_mask = density < (0.7 * base_threshold)
    delivered_no_need = int(np.sum(delivered & no_need_mask))

    # (Also useful sometimes:) Delivered but no action taken
    delivered_no_action = int(np.sum(delivered & (~actions)))

    # 4) Trust delta = (final mean trust across links) - (initial mean trust)
    # Grab first/last nan-safe values across links
    first_mean_trust = np.nanmean([
        _first_non_nan(state.trust_A1_A2_hist),
        _first_non_nan(state.trust_A2_A3_hist),
        _first_non_nan(state.trust_A3_A4_hist),
    ])
    last_mean_trust = np.nanmean([
        _last_non_nan(state.trust_A1_A2_hist),
        _last_non_nan(state.trust_A2_A3_hist),
        _last_non_nan(state.trust_A3_A4_hist),
    ])
    trust_delta = float(last_mean_trust - first_mean_trust)

    return dict(
        actions_count=actions_count,
        breach_steps=breach_steps,
        breach_events=breach_events,              # optional, see below
        delivered_no_need=delivered_no_need,      # your requested "no reason"
        delivered_no_action=delivered_no_action,  # useful diagnostic (optional)
        trust_delta=trust_delta,
    )

def compute_metrics(states: list, params: Params) -> dict:
    """
    Aggregate metrics across Monte-Carlo runs.
    We report means; you can add std/percentiles if desired.
    """
    per_run = [_per_state_metrics(s, params) for s in states]

    def mean_of(key):
        return float(np.mean([m[key] for m in per_run]))

    return dict(
        # Requested metrics (means across runs, rounded later)
        actions_mean=mean_of("actions_count"),
        breaches_mean=mean_of("breach_steps"),
        delivered_no_reason_mean=mean_of("delivered_no_need"),
        trust_delta_mean=mean_of("trust_delta"),

        # Optional diagnostics (you can include or drop in the table)
        breach_events_mean=mean_of("breach_events"),
        delivered_no_action_mean=mean_of("delivered_no_action"),
    )

def compute_per_run_metrics(states: list, params: Params):
    """Return raw (per-run) KPI values for significance tests."""
    per_run = [_per_state_metrics(s, params) for s in states]

    # Convert to arrays for convenience
    return {
        "actions":      np.array([m["actions_count"]        for m in per_run]),
        "breaches":     np.array([m["breach_steps"]         for m in per_run]),
        "no_reason":    np.array([m["delivered_no_need"]    for m in per_run]),
        "trust_delta":  np.array([m["trust_delta"]          for m in per_run])
    }


from scipy.stats import shapiro, levene, mannwhitneyu
import pandas as pd
import numpy as np


def run_assumption_tests(vals1, vals2):
    """Return dict with normality + variance test outcomes."""
    sw1 = shapiro(vals1)
    sw2 = shapiro(vals2)
    lev  = levene(vals1, vals2)

    return {
        "normal_1": sw1.pvalue > 0.05,
        "normal_2": sw2.pvalue > 0.05,
        "equal_var": lev.pvalue > 0.05,
        "p_sw1": sw1.pvalue,
        "p_sw2": sw2.pvalue,
        "p_lev": lev.pvalue,
    }


def mann_whitney_only(vals1, vals2):
    """Always run MWU as the inferential test."""
    stat = mannwhitneyu(vals1, vals2, alternative="two-sided")
    return stat.pvalue


def significance_tests_mann_whitney_only(metric_dict):
    """
    For every KPI and every scenario pair:
    - Run and store assumption tests (Shapiro, Levene)
    - Always use Mann–Whitney U for inference
    Returns:
        p-value tables, explanation tables
    """

    scenarios = list(metric_dict.keys())
    KPIS = ["actions", "breaches", "no_reason", "trust_delta"]

    out_pvals = {}
    out_assumptions = {}

    for kpi in KPIS:
        mat_p = pd.DataFrame(index=scenarios, columns=scenarios, dtype=float)
        mat_expl = pd.DataFrame(index=scenarios, columns=scenarios, dtype=object)

        for i, s1 in enumerate(scenarios):
            for j, s2 in enumerate(scenarios):
                if i >= j:
                    mat_p.loc[s1, s2] = np.nan
                    mat_expl.loc[s1, s2] = ""
                    continue

                vals1 = metric_dict[s1][kpi]
                vals2 = metric_dict[s2][kpi]

                # assumption tests (diagnostics only)
                A = run_assumption_tests(vals1, vals2)

                # ALWAYS use Mann–Whitney
                pval = mann_whitney_only(vals1, vals2)
                mat_p.loc[s1, s2] = pval

                mat_expl.loc[s1, s2] = (
                    f"Mann–Whitney U (chosen globally); "
                    f"normality p=({A['p_sw1']:.3f}, {A['p_sw2']:.3f}), "
                    f"variance p={A['p_lev']:.3f}"
                )

        out_pvals[kpi] = mat_p
        out_assumptions[kpi] = mat_expl

    return out_pvals, out_assumptions


def print_significance_summary_mw_only(sig_tables, sig_info, alpha=0.05):
    """
    Pretty console summary.
    """
    for kpi in sig_tables.keys():
        print(f"\n=========================\nKPI: {kpi}\n=========================")
        df = sig_tables[kpi]
        info = sig_info[kpi]

        for r in df.index:
            for c in df.columns:
                if pd.isna(df.loc[r, c]):
                    continue
                p = df.loc[r, c]
                explanation = info.loc[r, c]
                tag = "SIGNIFICANT" if p < alpha else "n.s."
                print(f"{r} vs {c}: {tag} (p={p:.4f}) — {explanation}")

def build_metrics_table(results_mc: dict, params: Params) -> pd.DataFrame:
    """
    Build the scenario outcome table with the requested columns:
      - actions taken
      - risk threshold breaches
      - delivered-but-no-reason
      - trust delta (end - init)
    Values are means across Monte-Carlo runs (rounded).
    """
    rows = []

    for name, states in results_mc.items():
        m = compute_metrics(states, params)
        rows.append({
            "scenario": name,
            "actions_taken": round(m["actions_mean"], 2),
            "risk_breaches": round(m["breaches_mean"], 2),
            "delivered_no_reason": round(m["delivered_no_reason_mean"], 2),
            "trust_delta": round(m["trust_delta_mean"], 3),
            # Optional diagnostics — uncomment if you want them in the table:
            # "breach_events": round(m["breach_events_mean"], 2),
            # "delivered_no_action": round(m["delivered_no_action_mean"], 2),
        })

    df = pd.DataFrame(rows).set_index("scenario")
    return df
# ============================================================
# Main
# ============================================================


if __name__ == "__main__":
    base_params = Params()

    results_mc = {}
    for scn in SCENARIOS:
        print(f"Running: {scn['name']}")
        results_mc[scn["name"]] = run_scenario_mc(
            scn, base_params, N_RUNS
        )

        # results = {}
        # for scn in SCENARIOS:
        #     print(f"Running: {scn['name']}")
        #     results[scn["name"]] = run_scenario(scn, base_params)

    from pathlib import Path
    from datetime import datetime
    import plotly.io as pio  # <-- This is the correct module for saving Plotly figs

    # --- Make a results directory ---
    run_dir = Path("results") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Build figures as you already do ---
    fig_density = density_comparison_figure_mc(results_mc, base_params)
    #fig_trust = trust_comparison_figure(results, base_params)

    fig_trust = trust_comparison_figure_mc(
        results_mc,
        base_params,
        conf_int=CONF_INT,
        band_style="fill",  # or "lines"
    )

    # --- Save Plotly figures (HTML recommended) ---
    pio.write_html(fig_trust, file=run_dir / "trust_mc.html", auto_open=False, include_plotlyjs="cdn")
    pio.write_html(fig_trust, file=run_dir / "trust_single.html", auto_open=False, include_plotlyjs="cdn")

    # --- Optional: save static images (requires `pip install -U kaleido`) ---
    # fig_density.write_image(run_dir / "density_mc.png", scale=2)  # PNG
    # fig_trust.write_image(run_dir / "trust_mc.png", scale=2)

    fig_density.write_image(
        run_dir / "density_mc.png",
        width=1400,
        height=550,
        scale=1
    )

    fig_trust.write_image(
        run_dir / "trust_mc.png",
        width=1400,
        height=550,
        scale=1
    )

    # --- Show figures interactively (optional) ---
    fig_density.show()
    fig_trust.show()

    # --- Save metrics table ---
    metrics_df = build_metrics_table(results_mc, base_params)
    metrics_df.to_csv(run_dir / "metrics_df_v1.csv", index=True)

    # === Significance testing (per-run metrics) ===
    metric_runs = {
        name: compute_per_run_metrics(states, base_params)
        for name, states in results_mc.items()
    }

    sig_tables, sig_info = significance_tests_mann_whitney_only(metric_runs)

    # Save outputs
    for kpi in sig_tables.keys():
        sig_tables[kpi].to_csv(run_dir / f"sig_{kpi}.csv")
        sig_info[kpi].to_csv(run_dir / f"sig_{kpi}_explanations.csv")

    print_significance_summary_mw_only(sig_tables, sig_info)

    print(f"Saved results to: {run_dir}")
    print(metrics_df)

    print(metrics_df)

# ============================================================
# Extra
# ============================================================

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

def per_metric_boxplots(metrics_runs: dict,
                        #title: str = "Distribution of KPIs across scenarios",
                        show_mean: bool = False) -> go.Figure:
    """
    2x2 grid of boxplots:
      - Per KPI (actions, breaches, no_reason, trust_delta)
      - Within each subplot: scenario-colored boxes grouped side-by-side
      - No x-axis labels (color + legend identify scenarios)
      - Box fills: scenario colors, fully opaque
      - Borders/whiskers/median lines: black for visibility
    """

    KPIS = [
        ("actions",     "Actions taken"),
        ("breaches",    "Risk breaches"),
        ("no_reason",   "Delivered (no reason)"),
        ("trust_delta", "Trust Δ"),
    ]

    scenarios = list(metrics_runs.keys())

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[k[1] for k in KPIS],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    def rc(idx):
        return (idx // 2) + 1, (idx % 2) + 1

    for idx, (kpi_key, kpi_label) in enumerate(KPIS):
        r, c = rc(idx)

        for scen in scenarios:
            vals = metrics_runs[scen][kpi_key]
            color = SCENARIO_COLORS.get(scen, "#666666")

            fig.add_trace(
                go.Box(
                    x=[scen] * len(vals),  # categorie per scenario (zorgt voor spacing)
                    y=vals,
                    name=scen,
                    legendgroup=scen,
                    showlegend=(idx == 0),

                    # === Styling zoals gevraagd ===
                    fillcolor=color,  # invulling in scenario-kleur
                    opacity=1.0,  # 100% dekkend
                    line=dict(color="black", width=1.6),  # zwarte omlijning + whiskers + median
                    marker=dict(color="black", size=4),  # outliers in zwart (optioneel)
                    width=0.5,  # box breedte (pas aan voor meer/ruimte)

                    # Box-eigenschappen
                    boxpoints="outliers",  # toon outliers
                    whiskerwidth=1.0,
                    boxmean=False,  # mean dot uit (academisch cleaner); True als je die wil

                    # Hover
                    hovertemplate=f"{scen}<br>{kpi_label}: %{{y:.2f}}<extra></extra>",
                ),
                row=r, col=c
            )

        # Hide x-axis ticks entirely
        fig.update_xaxes(
            showticklabels=False,
            tickvals=[],
            row=r,
            col=c,
        )

        # fig.update_yaxes(
        #     title_text=kpi_label,
        #     row=r, col=c
        # )

    fig.update_layout(
        #title=title,
        height=850,
        boxmode="group",
        boxgap=0.25,  # ruimte tussen boxen binnen groep (per subplot)
        boxgroupgap=0.30,  # extra lucht tussen groepen indien nodig
        margin=dict(l=60, r=260, t=90, b=70),  # extra rechter marge voor legenda
        hovermode="closest",
        font=dict(size=17),

        # === Legenda rechts, beter leesbaar ===
        legend=dict(
            orientation="v",  # verticaal stapelen
            yanchor="middle",
            y=0.5,  # middelhoog
            xanchor="left",
            x=1.02,  # naast de plot
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
            font=dict(size=15)
        ),
    )

    return fig

# def per_metric_grouped_bars_from_metrics(metrics_df: pd.DataFrame,
#                                          title: str = "Per‑metric grouped bars by scenario") -> go.Figure:
#     """
#     Builds a 2x2 grid: Actions, Breaches, Delivered-No-Reason, Trust Δ (bars per scenario).
#     Ensures the legend shows each scenario only once using legendgroup + showlegend=(idx==0).
#     """
#     df = metrics_df.copy()
#     # Ensure we have a "scenario" column for labels
#     if df.index.name is None or df.index.name.lower() != "scenario":
#         df = df.reset_index().rename(columns={"index": "scenario"})
#     else:
#         df = df.reset_index()
#
#     metrics = [
#         ("actions_taken",       "Actions"),
#         ("risk_breaches",       "Risk breaches"),
#         ("delivered_no_reason", "Delivered (no reason)"),
#         ("trust_delta",         "Trust Δ (end − init)"),
#     ]
#     scenarios = df["scenario"].tolist()
#
#     fig = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=[m[1] for m in metrics],
#         horizontal_spacing=0.12, vertical_spacing=0.15
#     )
#
#     def rc(idx):
#         return (idx // 2) + 1, (idx % 2) + 1
#
#     for idx, (col, label) in enumerate(metrics):
#         r, c = rc(idx)
#         for scen in scenarios:
#             val = float(df.loc[df["scenario"] == scen, col])
#             fig.add_trace(
#                 go.Bar(
#                     x=[label],  # single category per subplot
#                     y=[val],
#                     name=scen,  # keep same name for all subplots (legendgroup handles uniqueness)
#                     legendgroup=scen,                 # group all traces for this scenario
#                     showlegend=(idx == 0),            # only show in the first subplot
#                     marker_color=SCENARIO_COLORS.get(scen, "#666666"),
#                     hovertemplate=f"{scen}<br>{label}: %{{y:.2f}}<extra></extra>",
#                 ),
#                 row=r, col=c
#             )
#
#     # Optional: Add a zero line to emphasize sign for Trust Δ (global line)
#     fig.add_hline(y=0, line=dict(color="gray", dash="dot"))
#
#     fig.update_layout(
#         title=title,
#         barmode="group",
#         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#         margin=dict(l=60, r=60, t=90, b=60),
#         hovermode="closest"
#     )
#     return fig

#
# fig_grouped = per_metric_grouped_bars_from_metrics(metrics_df)
#
# fig_grouped.write_image(run_dir / "histo.png", scale=2)
#
#
# fig_grouped.write_image(
#     run_dir / "histo.png",
#     width=1400,
#     height=550,
#     scale=1
# )
#
#
# # Optional: also show
# #fig_tradeoff.show()
# fig_grouped.show()




fig_box = per_metric_boxplots(metric_runs)
fig_box.write_image(run_dir / "histo.png", width=1400, height=900)
fig_box.show()

