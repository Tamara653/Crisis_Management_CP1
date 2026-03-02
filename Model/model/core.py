# ============================================================
#  model/core.py

"""
core.py — Main simulation model for multi-agent human–AI teaming.

This module implements a discrete-time simulation environment with:
- A1: Sensor / forecasting agent performing noisy lookahead measurements
- A2: Analyst generating recommendations based on A1 measurements
- A3: Communications relay agent (transmission reliability + latency)
- A4: Decider performing actions that modify the system’s density trajectory

The simulation also includes:
- Multiple baseline modes (sinusoidal or realistic event-like patterns)
- Trust-mapped reliability and latency adjustments
- Action effects with dampening dynamics
- History tracking for measurements, actions, and trust

The Params dataclass contains all simulation hyperparameters.
SimState stores all evolving values for each timestep.
"""

# ============================================================

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

#from OLD.crowd_management import threshold


# =============================
# Parameters
# =============================

@dataclass
class Params:

    """
    Configuration and tunable parameters governing the simulation.

    This includes:
    - Agent reliabilities
    - Agent latencies
    - Decision thresholds
    - Baseline generation parameters
    - Trust update parameters
    - Behavior of actions and their effects
    - Sensor noise modeling
    - Lookahead behavior
    """


    # --- agent reliabilities / latencies ---
    A1_reliability: float = 0.95
    A1_latency: int = 0
    A2_reliability: float = 0.80
    A2_latency: int = 1
    threshold_analyst: float = 0.85
    A3_reliability: float = 0.90
    A3_latency: int = 2

    A4_reliability: float = 0.90

    # NEW: Decider latency (real)
    A4_latency: int = 0

    # Action parameters
    action_reduction: float = 1.2
    action_cost: float = 1.0
    threshold_decider: float = 1.1

    # --- environment ---
    steps: int = 81
    minutes_per_step: int = 1
    threshold: float = 4.0

    # Synthetic sinusoidal parameters
    base_level: float = 0.5
    amplitude: float = 1.25
    noise_sigma: float = 0.5

    # NEW: Baseline mode ("sinusoidal" or "realistic")
    baseline_mode: str = "realistic"

    # Dampening after action
    damp_steps: int = 3
    dampen_alpha: float = 0.4

    # Columns for DF-based mode
    baseline_column: str = "true_value"
    measure_column: str = "pred_p50"

    # Action effect decomposition
    use_dynamic_action_effect: bool = True
    action_effect_w_current: float = 0.10
    action_effect_w_incoming: float = 0.9

    # --- Trust mapping ---
    seed: int = 43
    trust_A1_A2: float = 0.0
    trust_A2_A3: float = 0.0
    trust_A3_A4: float = 0.0
    k_rel: float = 0.10
    k_lat: float = 1.0
    gamma_threshold: float = 0.15
    eta_up: float = 0.05
    eta_down: float = 0.08

    # --- A1 lookahead ---
    sensor_lookahead_steps: int = 12

    # --- A1 noise controls ---
    A1_sigma0: float = 0.5
    A1_alpha_h: float = 0.3
    A1_gamma_r: float = 0.7
    A1_beta_peak: float = 0.6
    A1_peak_lo_q: float = 0.20
    A1_peak_hi_q: float = 0.80
    A1_acc_k: float = 1.0


# =============================
# Simulation State
# =============================

@dataclass
class SimState:

    """
    Represents the full simulation state across all timesteps.

    Tracks:
    - Sensor measurements + accuracy
    - Analyst recommendations
    - Communication successes
    - Decider measurements + actions
    - Action success and effects
    - Trust evolution and effective reliabilities/latencies
    - Baseline and density states
    """

    t: int
    baseline: np.ndarray
    density: np.ndarray
    sensor_meas: np.ndarray
    sensor_acc: np.ndarray
    analyst_recommend: np.ndarray
    comms_success: np.ndarray
    decider_meas: np.ndarray
    decider_meas_acc: np.ndarray
    decider_action: np.ndarray
    action_success: np.ndarray
    action_effect: np.ndarray
    dampen_applied: np.ndarray

    trust_A1_A2: float
    trust_A2_A3: float
    trust_A3_A4: float

    trust_A1_A2_hist: np.ndarray
    trust_A2_A3_hist: np.ndarray
    trust_A3_A4_hist: np.ndarray
    eff_rel_A2_hist: np.ndarray
    eff_lat_A2_hist: np.ndarray
    eff_rel_A3_hist: np.ndarray
    eff_lat_A3_hist: np.ndarray
    eff_rel_A4_act_hist: np.ndarray

    T_MINUTES: np.ndarray
    T: int

    df_measure_df: Optional[pd.DataFrame] = None
    df_measure_col: Optional[str] = None
    df_start_idx: int = 0
    threshold: Optional[float] = None

    baseline_p_lo: float = 0.0
    baseline_p_hi: float = 0.0


# ============================================================
# Baseline generators (sinusoidal + realistic)
# ============================================================

def generate_realistic_baseline(
    T=81,
    base_level=0.5,
    peak_density=2.5,
    n_pulses=5,
    pulse_width=5,
    pulse_height=0.6,
    noise_low=0.05,
    noise_high=0.45,
    seed=42
):

    """
    Generate a synthetic event-like baseline with:
    - Rise → plateau → decay
    - Random Gaussian-like pulses
    - Heteroscedastic noise that increases during peak event periods

    Returns:
        np.ndarray of shape (T,)
    """

    rng = np.random.RandomState(seed)
    t = np.arange(T)

    # Piecewise trend
    arrival_len = int(0.25 * T)
    plateau_len = int(0.50 * T)
    decay_len = T - arrival_len - plateau_len

    arrival = np.linspace(base_level, peak_density, arrival_len)
    plateau = np.full(plateau_len, peak_density)
    decay = np.linspace(peak_density, base_level, decay_len)

    B = np.concatenate([arrival, plateau, decay])

    # Pulses
    P = np.zeros(T)
    centers = rng.randint(int(0.1*T), int(0.9*T), size=n_pulses)
    for c in centers:
        P += pulse_height * np.exp(-0.5 * ((t - c) / pulse_width)**2)

    # Heteroscedastic noise
    density_noiseless = B + P
    rel = (density_noiseless - base_level) / (peak_density - base_level + 1e-6)
    rel = np.clip(rel, 0, 1)
    sigma = noise_low + rel * (noise_high - noise_low)
    eps = rng.normal(0, sigma)

    return np.clip(density_noiseless + eps, 0, None)


def generate_sinusoidal_baseline(steps, base_level, amplitude, noise_sigma, seed):
    """
        Generate a smooth sinusoidal baseline with additive Gaussian noise.

        Intended to simulate diurnal or periodic dynamics.

        Args:
            steps: total simulation steps
            base_level: vertical offset
            amplitude: sinusoidal amplitude
            noise_sigma: Gaussian measurement noise
            seed: RNG seed

        Returns:
            np.ndarray of non-negative values
        """

    rng = np.random.RandomState(seed)
    t = np.arange(steps, dtype=float)
    baseline = (
        base_level
        + amplitude * np.sin(2 * np.pi * t / 24.0)
        + rng.normal(0, noise_sigma, steps)
    )
    return np.clip(baseline, 0.0, None)


# ============================================================
# Baseline Routing
# ============================================================

def make_baseline(params: Params,
                  baseline_df: Optional[pd.DataFrame] = None,
                  column_name: Optional[str] = None) -> np.ndarray:
    """
        Return the baseline series used by the simulation.

        Priority:
        1. If a DataFrame and column are provided: sample a contiguous slice.
        2. Else if baseline_mode == 'sinusoidal': generate sinusoidal baseline.
        3. Else: generate realistic/event-based baseline.

        Returns:
            baseline array of length params.steps
        """

    steps = int(params.steps)
    col = column_name or params.baseline_column
    rng = np.random.RandomState(params.seed)

    # DataFrame-based baseline
    if baseline_df is not None and col in baseline_df.columns:
        full_series = baseline_df[col].to_numpy(dtype=float)
        start_i = rng.randint(0, max(1, len(full_series) - steps)) #random chunk if not full_series
        return full_series[start_i:start_i + steps]

    # Synthetic modes
    if params.baseline_mode == "sinusoidal":
        return generate_sinusoidal_baseline(
            steps,
            params.base_level,
            params.amplitude,
            params.noise_sigma,
            params.seed
        )

    # Default: event-realistic
    return generate_realistic_baseline(
        T=params.steps,
        base_level=params.base_level,
        peak_density=params.threshold,
        seed=params.seed
    )


# ============================================================
# A1 Measurement with Noise based on LoA accuracy
# ============================================================

def sensor_noise_std(true_val, reliability, horizon_steps,
                     sigma0, alpha_h, gamma_r, beta_peak, p_lo, p_hi):
    """
    Compute the effective standard deviation of the A1 sensor's noisy measurement.

    Noise increases based on three interacting components:

    1) Reliability factor  f_r
       - Lower reliability → larger noise
       - Nonlinear exponent gamma_r controls how sharply noise grows as reliability drops

    2) Horizon (lookahead) factor  f_h
       - Longer lookahead steps increase uncertainty
       - Controlled by alpha_h (slope of increase per horizon step)

    3) Peak-awareness factor  f_p
       - If the true density lies near extreme quantiles (very low or very high),
         noise is inflated using beta_peak
       - Models instability of sensing at extreme system states

    Final formula (clamped at a minimum of 1e-6):
        sigma = sigma0 * f_r * f_h * f_p

    Args:
        true_val (float):       The true future baseline value being sensed.
        reliability (float):    Sensor reliability in [0,1], affects f_r.
        horizon_steps (int):    Lookahead length; influences f_h.
        sigma0 (float):         Base noise level when reliability = 1 and horizon = 0.
        alpha_h (float):        Scaling coefficient for horizon noise.
        gamma_r (float):        Exponent for the reliability-based noise term.
        beta_peak (float):      Peak-region amplification factor.
        p_lo (float):           Lower quantile threshold for peak-awareness.
        p_hi (float):           Upper quantile threshold for peak-awareness.

    Returns:
        float: Standard deviation used when generating the Gaussian sensor noise.
    """

    eps = 1e-6  # small constant to avoid division by zero or zero-powers

    # ------------------------------------------------------------
    # 1) Reliability factor: noise increases when reliability drops
    # ------------------------------------------------------------
    # Convert reliability into a "distance from perfect reliability"
    r = float(np.clip(reliability, 0, 1))
    # f_r = (1 - r)^gamma_r, but clamped to avoid zero
    f_r = max(eps, (1 - r)) ** float(gamma_r)

    # ------------------------------------------------------------
    # 2) Horizon factor: uncertainty grows with lookahead distance
    # ------------------------------------------------------------
    # f_h = 1 + alpha_h * horizon
    x = max(0, int(horizon_steps))
    f_h = 1 + float(alpha_h) * x

    # ------------------------------------------------------------
    # 3) Peak-awareness factor: noise increases at extreme values
    # ------------------------------------------------------------
    # If the signal is near either tail of the baseline distribution,
    # the sensor becomes less reliable (e.g., sparse data, nonlinearities).
    if true_val <= p_lo or true_val >= p_hi:
        f_p = (1 + beta_peak)
    else:
        f_p = 1.0

    # ------------------------------------------------------------
    # Combine components to produce final noise std
    # ------------------------------------------------------------
    sigma = sigma0 * f_r * f_h * f_p

    # Prevent numerical issues from zero or negative values
    return max(sigma, 1e-6)


# ============================================================
# Trust-mapped reliability and latency
# ============================================================

def eff_rel(base_rel, trust, k_rel):
    """Return reliability adjusted by trust level."""
    return float(np.clip(base_rel + k_rel * trust, 0.0, 0.999))


def eff_lat(base_lat, trust, k_lat):
    """Return latency reduced or increased by trust."""
    return int(max(0, round(base_lat - k_lat * trust)))


# ============================================================
# init_state
# ============================================================

def init_state(params: Params, baseline_df: Optional[pd.DataFrame] = None) -> SimState:

    """
    Initialize a fresh simulation state with all histories and arrays set.

    Returns:
        SimState instance pre-populated for timestep 0.
    """

    steps = int(params.steps)
    minutes = int(params.minutes_per_step)
    X = np.arange(0, steps * minutes, minutes)

    rng = np.random.RandomState(params.seed)

    baseline = make_baseline(params, baseline_df, params.baseline_column)
    T = len(baseline)

    p_lo = float(np.quantile(baseline, params.A1_peak_lo_q))
    p_hi = float(np.quantile(baseline, params.A1_peak_hi_q))

    return SimState(
        t=0,
        baseline=baseline.copy(),
        density=baseline.copy(),
        sensor_meas=np.full(T, np.nan),
        sensor_acc=np.zeros(T, bool),
        analyst_recommend=np.zeros(T, bool),
        comms_success=np.zeros(T, bool),
        decider_meas=np.full(T, np.nan),
        decider_meas_acc=np.zeros(T, bool),
        decider_action=np.zeros(T, bool),
        action_success=np.zeros(T, bool),
        action_effect=np.zeros(T),
        dampen_applied=np.zeros(T, bool),
        trust_A1_A2=params.trust_A1_A2,
        trust_A2_A3=params.trust_A2_A3,
        trust_A3_A4=params.trust_A3_A4,
        trust_A1_A2_hist=np.full(T, np.nan),
        trust_A2_A3_hist=np.full(T, np.nan),
        trust_A3_A4_hist=np.full(T, np.nan),
        eff_rel_A2_hist=np.full(T, np.nan),
        eff_lat_A2_hist=np.zeros(T, int),
        eff_rel_A3_hist=np.full(T, np.nan),
        eff_lat_A3_hist=np.zeros(T, int),
        eff_rel_A4_act_hist=np.full(T, np.nan),
        T_MINUTES=X,
        T=T,
        df_measure_df=baseline_df,
        df_measure_col=params.measure_column if baseline_df is not None else None,
        baseline_p_lo=p_lo,
        baseline_p_hi=p_hi,
        threshold=params.threshold
    )


# ============================================================
# step_once
# ============================================================

def step_once(state: SimState, params: Params) -> SimState:
    """
    Advance the simulation by one timestep.

    Stages:
    1. A1 performs a noisy lookahead measurement.
    2. A2 evaluates the measurement and may recommend action.
    3. A3 may successfully transmit that recommendation.
    4. A4 decides whether to act based on comms + its own measurement.
    5. If action triggers: apply immediate effect and follow-up dampening.
    6. Update trust among A1→A2, A2→A3, A3→A4 based on outcomes.

    Returns:
        Updated SimState after one simulation step.
    """

    T = state.T
    t = state.t
    if t >= T:
        # Already at or beyond final timestep — no update needed
        return state

    # Local RNG seeded per timestep for reproducibility
    rng = np.random.RandomState(params.seed + t)

    # ============================================================
    # 0) Compute trust‑adjusted reliabilities and latencies
    # ============================================================

    # Analyst reliability adjusted by trust(A1→A2)
    A2_rel_eff = eff_rel(params.A2_reliability, state.trust_A1_A2, params.k_rel)
    # Analyst latency adjusted by trust(A1→A2)
    A2_lat_eff = eff_lat(params.A2_latency, state.trust_A1_A2, params.k_lat)

    # Comms reliability adjusted by trust(A2→A3)
    A3_rel_eff = eff_rel(params.A3_reliability, state.trust_A2_A3, params.k_rel)
    # Comms latency adjusted by trust(A2→A3)
    A3_lat_eff = eff_lat(params.A3_latency, state.trust_A2_A3, params.k_lat)

    # Decider (A4) reliability for *actions* adjusted by trust(A3→A4)
    A4_rel_eff = eff_rel(params.A4_reliability, state.trust_A3_A4, params.k_rel)

    # Store history for analysis/plotting later
    state.trust_A1_A2_hist[t] = state.trust_A1_A2
    state.trust_A2_A3_hist[t] = state.trust_A2_A3
    state.trust_A3_A4_hist[t] = state.trust_A3_A4
    state.eff_rel_A2_hist[t] = A2_rel_eff
    state.eff_lat_A2_hist[t] = A2_lat_eff
    state.eff_rel_A3_hist[t] = A3_rel_eff
    state.eff_lat_A3_hist[t] = A3_lat_eff
    state.eff_rel_A4_act_hist[t] = A4_rel_eff

    # ============================================================
    # 1) A1: Lookahead measurement
    # ============================================================

    # Lookahead step (e.g., forecasting future state)
    x = int(params.sensor_lookahead_steps)
    sense_t = min(t + x, T - 1)

    # True future value used to generate noisy measurement
    a1_true = float(state.baseline[sense_t])

    # Compute sensor noise std based on reliability + peak quantiles
    sigma_eff = sensor_noise_std(
        a1_true, params.A1_reliability, x,
        params.A1_sigma0, params.A1_alpha_h,
        params.A1_gamma_r, params.A1_beta_peak,
        state.baseline_p_lo, state.baseline_p_hi
    )

    # Generate noisy measurement
    meas = rng.normal(a1_true, sigma_eff)
    meas = max(0, meas)  # Density cannot go negative

    # Accuracy flag: whether the measurement falls within ±k·sigma
    acc = (abs(meas - a1_true) <= params.A1_acc_k * sigma_eff)

    # Store measurement and accuracy
    state.sensor_meas[t] = meas
    state.sensor_acc[t] = acc

    # ============================================================
    # 2) A2: Analyst recommendation logic
    # ============================================================

    # A2 receives A1 measurement delayed by A1 latency
    rec_available_t = t - params.A1_latency

    if rec_available_t >= 0:
        meas_A2 = state.sensor_meas[rec_available_t]
        # Compare against analyst threshold
        trigger = meas_A2 >= params.threshold_analyst * params.threshold

        # A2 may fail due to reliability — unreliable A2 may invert the signal
        rec = trigger if (rng.rand() <= A2_rel_eff) else (not trigger)

        # Recommendation is scheduled after A2’s effective latency
        issue_t = t + A2_lat_eff
        if 0 <= issue_t < T:
            state.analyst_recommend[issue_t] = bool(rec)

    # ============================================================
    # 3) A3: Communications relay
    # ============================================================
    # At time t, a recommendation is 'ready' for relay if it was issued A3_lat_eff steps ago.
    comms_ok = False
    rec_ready_t = t - A3_lat_eff
    has_rec = (rec_ready_t >= 0) and bool(state.analyst_recommend[rec_ready_t])

    if has_rec:
        comms_ok = (rng.rand() <= A3_rel_eff)
    state.comms_success[t] = comms_ok


    # ============================================================
    # 4) A4: Decider logic — may choose to intervene
    # ============================================================
    want_act = False
    if comms_ok and has_rec:
    # Early/preventive actions are driven by the upstream recommendation (foresight path)
        want_act = True
    # Local “panic” path: act only when near-critical without a recommendation
    elif state.density[t] >= params.threshold_decider * params.threshold + params.gamma_threshold * state.trust_A3_A4:
        want_act = True

    # Schedule the action after A4 latency
    act_time = t + params.A4_latency
    if want_act and act_time < T:
        state.decider_action[act_time] = True
        # Action may fail with probability (1−reliability)
        state.action_success[act_time] = (rng.rand() <= A4_rel_eff)

    # ============================================================
    # 5) Execute action effects + dampening if action fires
    # ============================================================

    if state.decider_action[t] and state.action_success[t] and t+1 < T:

        # Compute the action effect using decomposition:
        true_curr = float(state.baseline[t])
        true_next = float(state.baseline[t+1])
        incoming_delta = true_next - true_curr

        effect = (
            params.action_effect_w_current * true_curr +
            params.action_effect_w_incoming * incoming_delta
        )

        # Clamp effect so it cannot overshoot
        effect = max(0, min(effect, state.density[t+1]))
        state.action_effect[t] = effect

        # Apply immediate effect (reduce next timestep density)
        state.density[t+1] = max(0, state.density[t+1] - effect)

        # Apply dampening over next K timesteps
        for k in range(1, params.damp_steps+1):
            future = t + k
            if future < T:
                alpha = params.dampen_alpha
                # Weighted mixture of previous density
                state.density[future] = max(
                    0,
                    alpha * state.density[future] + (1-alpha) * state.density[future-1]
                )
                state.dampen_applied[future] = True

    # ============================================================
    # 6) Trust Update Logic
    # ============================================================

    def clamp(x, d):
        """Clamp trust updates between −1 and +1."""
        return float(np.clip(x + d, -1, 1))

    # Case A: An action was taken based on recommendation
    if state.decider_action[t] and state.density[t]<= params.threshold_decider * params.threshold and t+1 < T:

        K = 5  # ahead check
        # end index is exclusive; clamp to T
        end_idx = min(state.T, t + 1 + K)

        # Slice the available window [t+1, end_idx)
        window = state.density[t + 1: end_idx]

        # If the slice is empty (e.g., t == T-1), fall back to something sensible
        if window.size > 0:
            d_next = float(np.max(window))
        else:
            d_next = float(state.density[t])  # or np.nan, depending on your desig

        # "Good" outcome: density ends up near threshold band
        good = (params.threshold*0.7 <= d_next < params.threshold*1.4)
        # "Bad" outcome: density too low or too high
        bad  = (d_next < params.threshold*0.7) or (d_next >= params.threshold*1.2)

        if good:
            # Positive trust reinforcement
            state.trust_A1_A2 = clamp(state.trust_A1_A2, params.eta_up)
            state.trust_A2_A3 = clamp(state.trust_A2_A3, params.eta_up)
            state.trust_A3_A4 = clamp(state.trust_A3_A4, params.eta_up)

        elif bad:
            # Negative trust update (penalty)
            state.trust_A1_A2 = clamp(state.trust_A1_A2, -params.eta_down)
            state.trust_A2_A3 = clamp(state.trust_A2_A3, -params.eta_down)
            state.trust_A3_A4 = clamp(state.trust_A3_A4, -params.eta_down)

    # Case B: No action taken, BUT density goes above threshold → failed pipeline
    if not state.decider_action[t] and t+1 < T and state.density[t+1] >= params.threshold:
        state.trust_A1_A2 = clamp(state.trust_A1_A2, -params.eta_down)
        state.trust_A2_A3 = clamp(state.trust_A2_A3, -params.eta_down)
        state.trust_A3_A4 = clamp(state.trust_A3_A4, -params.eta_down)

    # Case C: fragmented action taken
    if state.decider_action[t] and state.density[t]>= params.threshold_decider * params.threshold and t+1 < T:
        state.trust_A3_A4 = clamp(state.trust_A3_A4, -params.eta_down)

    # Advance timestep
    state.t = t + 1
    return state