"""Umberger metabolic-rate post-processing for Biosym results.

The equations are translated from BioMAC-Sim-Toolbox:
`@Model/getErate_Umberger.m` and `@Model/getEratec_umberger.m`. Current BioSym
Hill models store neural excitation directly; legacy trajectories without that
state remain supported by reconstructing it from activation dynamics.
"""

from __future__ import annotations

import inspect

import numpy as np

from src.metabolic_common import (
    Formulation,
    MetabolicRateResult,
    as_flat_array,
    as_nodes_by_muscles,
    positive_part_smooth,
    summarize_metabolic_rate,
)


def _get_umberger_constants(biosym_model) -> dict[str, np.ndarray]:
    actuator = biosym_model.actuator_model
    muscle_constants = actuator.muscle_constants
    return {
        "lceopt": as_flat_array(muscle_constants["lceopt"]),
        "width": as_flat_array(muscle_constants["width"]),
        "fmax": as_flat_array(muscle_constants["fmax"]),
        "vmax": as_flat_array(muscle_constants["vmax"]),
        "ft": np.asarray(
            [
                float(muscle.get("ft", muscle.get("FT", 0.5)))
                for muscle in actuator.muscles_dict
            ],
            dtype=float,
        ),
    }


def estimate_excitation_from_activation(
    activation: np.ndarray,
    *,
    dt: float,
    tact: np.ndarray,
    tdeact: np.ndarray,
    edge_order: int = 1,
) -> np.ndarray:
    """Estimate excitation by inverting BioMAC activation dynamics.

    BioMAC documents activation dynamics as ``da/dt = rate * (u - a)``.
    This helper uses ``Tact`` when activation rises and ``Tdeact`` when it
    falls, then clips the reconstructed excitation into ``[0, 1]``.
    """

    activation = np.asarray(activation, dtype=float)
    da_dt = np.gradient(activation, float(dt), axis=0, edge_order=edge_order)
    tau = np.where(da_dt >= 0.0, tact, tdeact)
    return np.clip(activation + da_dt * tau, 0.0, 1.0)


def derive_fiber_velocity(l_ce: np.ndarray, duration: float) -> np.ndarray:
    """Reproduce BioSym's backward-Euler ``Lce_dot`` for saved trajectories."""

    l_ce = np.asarray(l_ce, dtype=float)
    if len(l_ce) < 2:
        return np.zeros_like(l_ce)
    dt = float(duration) / (len(l_ce) - 1)
    if dt <= 0.0:
        raise ValueError("Trajectory duration must be positive to derive Lce_dot.")
    velocity = np.empty_like(l_ce)
    velocity[1:] = np.diff(l_ce, axis=0) / dt
    # Saved periodic trajectories duplicate node 0 at their final sample, so
    # the predecessor of node 0 is the penultimate sample.
    predecessor = l_ce[-2] if len(l_ce) > 2 else l_ce[-1]
    velocity[0] = (l_ce[0] - predecessor) / dt
    return velocity


def umberger_original_rate(
    f_ce: np.ndarray,
    excitation: np.ndarray,
    activation: np.ndarray,
    l_ce: np.ndarray,
    v_ce: np.ndarray,
    *,
    lceopt: np.ndarray,
    width: np.ndarray,
    fmax: np.ndarray,
    vmax: np.ndarray,
    ft: np.ndarray,
    aerobic_factor: float = 1.5,
) -> np.ndarray:
    """Umberger 2010 rate translated from BioMAC ``getErate_Umberger.m``."""

    rho = 1059.7
    sigma = 250e3

    a_eff = np.where(excitation > activation, excitation, 0.5 * (excitation + activation))
    f_iso = np.exp(-((l_ce - 1.0) ** 2) / (width**2))
    muscle_mass = (fmax / sigma) * rho * lceopt

    v_cemax_ft = vmax
    v_cemax_st = v_cemax_ft / 2.5

    nh_am = np.where(activation <= (1.0 - ft), 25.0, 128.0 * ft + 25.0)
    a_am = a_eff**0.6
    h_am_short = nh_am * a_am * aerobic_factor
    h_am_long = (0.4 * nh_am + 0.6 * nh_am * f_iso) * a_am * aerobic_factor
    h_am = np.where(l_ce <= 1.0, h_am_short, h_am_long)

    alpha_st = 100.0 / v_cemax_st
    alpha_ft = (153.0 / v_cemax_ft) * (activation > 1.0 - ft)
    alpha_l = 0.3 * alpha_st

    nh_sl_shortening_limit = alpha_st * v_cemax_st < -alpha_st * v_ce * (1.0 - ft)
    nh_sl_shortening = (
        100.0 * nh_sl_shortening_limit
        - alpha_st * v_ce * (1.0 - ft) * (~nh_sl_shortening_limit)
        - alpha_ft * v_ce * ft
    )
    nh_sl = np.where(v_ce > 0.0, alpha_l * v_ce, nh_sl_shortening)

    a_sl = np.where(v_ce > 0.0, a_eff, a_eff**2.0)
    h_sl = nh_sl * a_sl * aerobic_factor
    h_sl = np.where(l_ce > 1.0, h_sl * f_iso, h_sl)

    w_ce = np.maximum(0.0, -f_ce * v_ce * lceopt / muscle_mass)
    return (h_am + h_sl + w_ce) * muscle_mass


def umberger_continuous_rate(
    f_ce: np.ndarray,
    excitation: np.ndarray,
    activation: np.ndarray,
    l_ce: np.ndarray,
    v_ce: np.ndarray,
    *,
    lceopt: np.ndarray,
    width: np.ndarray,
    fmax: np.ndarray,
    vmax: np.ndarray,
    ft: np.ndarray,
    epsilon: float = 1e-6,
    aerobic_factor: float = 1.5,
) -> np.ndarray:
    """Continuous Umberger rate translated from BioMAC ``getEratec_umberger.m``."""

    rho = 1059.7
    sigma = 250e3
    muscle_mass = (fmax / sigma) * lceopt * rho

    acst = 0.5 * (activation + excitation)
    a_eff = excitation + 0.5 * (
        (acst - excitation) + np.sqrt((acst - excitation) ** 2 + epsilon**2)
    )

    f_iso = np.exp(-((l_ce - 1.0) ** 2) / (width**2))
    f_iso1 = np.where(l_ce <= 1.0, 1.0, f_iso)

    v_cemax_ft = vmax
    v_cemax_st = v_cemax_ft / 2.5
    v_ce_l = 0.5 * (v_ce + np.sqrt(v_ce**2 + epsilon**2))
    v_ce_s = 0.5 * (v_ce - np.sqrt((-v_ce) ** 2 + epsilon**2))

    nh_am = np.where(activation <= (1.0 - ft), 25.0, 128.0 * ft + 25.0)
    a_am = np.real(a_eff**0.6)
    h_am = (0.4 * nh_am + 0.6 * nh_am * f_iso1) * a_am * aerobic_factor

    alpha_st = 100.0 / v_cemax_st
    alpha_ft = (153.0 / v_cemax_ft) * (activation > 1.0 - ft)
    alpha_l = 0.3 * alpha_st
    nh_sl = alpha_l * v_ce_l + (
        100.0 * (alpha_st * v_cemax_st < -alpha_st * v_ce_s * (1.0 - ft))
        - alpha_st
        * v_ce_s
        * (1.0 - ft)
        * (alpha_st * v_cemax_st > -alpha_st * v_ce_s * (1.0 - ft))
        - alpha_ft * v_ce_s * ft
    )
    h_sl = nh_sl * (a_eff**2.0) * aerobic_factor * f_iso1

    w_cebar = -f_ce * v_ce * lceopt / muscle_mass
    w_ce = positive_part_smooth(w_cebar, epsilon)
    return (h_am + h_sl + w_ce) * muscle_mass


def compute_umberger_metabolic_rate(
    biosym_model,
    trajectory,
    *,
    formulation: Formulation = "original",
    epsilon: float = 1e-6,
    excitation: np.ndarray | None = None,
    duration: float | None = None,
    dt: float | None = None,
    body_mass: float | None = None,
    speed: float | None = None,
    gravity: float = 9.81,
    match_toolbox_node_average: bool = True,
) -> MetabolicRateResult:
    """Compute Umberger metabolic rate for a Biosym trajectory."""

    actuator = biosym_model.actuator_model
    n_muscles = actuator.get_n_actuators()

    actuator_states = np.asarray(trajectory.states.actuator_model, dtype=float)
    l_ce = actuator_states[:, np.asarray(actuator.idx["Lce"], dtype=int)]
    if "Lce_dot" in actuator.idx:
        v_ce = actuator_states[:, np.asarray(actuator.idx["Lce_dot"], dtype=int)]
    else:
        if duration is None:
            raise ValueError("Pass duration to derive Lce_dot for this BioSym model.")
        v_ce = derive_fiber_velocity(l_ce, duration)
    activation = actuator_states[:, np.asarray(actuator.idx["a"], dtype=int)]
    # IPOPT may return values a few e-9 below the nominal nonnegative bound.
    # Fractional activation powers in Umberger are undefined for those tiny
    # negative residuals, so enforce the physiological lower bound here.
    activation = np.maximum(activation, 0.0)

    muscle_equation_parameters = inspect.signature(actuator.muscle_equations).parameters
    if "L_ce_dot" in muscle_equation_parameters:
        f_ce, _, _ = actuator.muscle_equations(
            trajectory.states, v_ce, trajectory.constants, biosym_model
        )
    else:
        f_ce, _, _ = actuator.muscle_equations(
            trajectory.states, trajectory.constants, biosym_model
        )
    f_ce = as_nodes_by_muscles(f_ce, n_muscles)

    constants = _get_umberger_constants(biosym_model)
    if excitation is None and "e" in actuator.idx:
        excitation = actuator_states[:, np.asarray(actuator.idx["e"], dtype=int)]
    elif excitation is None:
        if dt is None:
            if duration is None:
                raise ValueError("Pass either `duration` or `dt` to reconstruct excitation.")
            dt = float(duration) / (activation.shape[0] - 1)
        muscle_constants = actuator.muscle_constants
        excitation = estimate_excitation_from_activation(
            activation,
            dt=float(dt),
            tact=as_flat_array(muscle_constants["Tact"]),
            tdeact=as_flat_array(muscle_constants["Tdeact"]),
        )
    else:
        excitation = np.asarray(excitation, dtype=float)
    excitation = np.maximum(excitation, 0.0)

    if formulation == "original":
        by_muscle = umberger_original_rate(
            f_ce,
            excitation,
            activation,
            l_ce,
            v_ce,
            **constants,
        )
    elif formulation == "continuous":
        by_muscle = umberger_continuous_rate(
            f_ce,
            excitation,
            activation,
            l_ce,
            v_ce,
            epsilon=epsilon,
            **constants,
        )
    else:
        raise ValueError(
            f"Unknown formulation {formulation!r}; use 'continuous' or 'original'."
        )

    return summarize_metabolic_rate(
        biosym_model,
        by_muscle,
        list(actuator.names),
        duration=duration,
        body_mass=body_mass,
        speed=speed,
        gravity=gravity,
        match_toolbox_node_average=match_toolbox_node_average,
        include_resting_rate=True,
    )
