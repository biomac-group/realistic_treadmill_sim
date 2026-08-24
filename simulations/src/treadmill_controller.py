from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d


AP_GRF_FILTER_CUTOFF_HZ = 10.0
POST_HS_INTERPOLATION_SECONDS = 0.050
CUBIC_ANCHORS_BEFORE = 5
CUBIC_ANCHORS_AFTER = 6


@lru_cache(maxsize=None)
def _cubic_interp_weights(gap):
    """Linear weights reproducing scipy interp1d(kind='cubic') for fixed nodes."""
    anchor_x = np.r_[
        np.arange(-CUBIC_ANCHORS_BEFORE, 0),
        np.arange(gap, gap + CUBIC_ANCHORS_AFTER),
    ].astype(float)
    identity = np.eye(len(anchor_x))
    return np.asarray(
        interp1d(anchor_x, identity, kind="cubic", axis=0, assume_sorted=True)(
            np.arange(gap, dtype=float)
        )
    )


def cubic_bridge_post_hs(values, heel_strike_index, interpolation_samples):
    """Cubic interpolation of the full HS-to-edge gap using four anchors."""
    values = jnp.asarray(values)
    if interpolation_samples <= 0:
        return values
    n = values.shape[0]
    gap = int(interpolation_samples)
    anchor_offsets = np.r_[
        np.arange(-CUBIC_ANCHORS_BEFORE, 0),
        np.arange(gap, gap + CUBIC_ANCHORS_AFTER),
    ]
    anchor_indices = (int(heel_strike_index) + anchor_offsets) % n
    anchor_y = values[anchor_indices]
    bridged = jnp.asarray(_cubic_interp_weights(gap), dtype=values.dtype) @ anchor_y
    offsets = jnp.arange(gap, dtype=jnp.int32)
    indices = (int(heel_strike_index) + offsets.astype(jnp.int32)) % n
    return values.at[indices].set(bridged)


# Backward-compatible name for older analysis imports.
pchip_bridge_post_hs = cubic_bridge_post_hs


def periodic_butterworth_filtfilt(values, dt, cutoff_hz=AP_GRF_FILTER_CUTOFF_HZ):
    """Second-order Butterworth filtfilt response for a periodic trajectory.

    The controller is part of a cyclic, trajectory-level OCP. Applying the
    forward-backward transfer function in the frequency domain is equivalent
    to steady-periodic ``filtfilt``: it has zero phase and no artificial
    initial/final condition at the gait-cycle boundary. The implementation is
    fully JAX differentiable, unlike calling scipy.signal.filtfilt here.
    """
    values = jnp.asarray(values)
    n = values.shape[0]
    k = jnp.tan(jnp.pi * jnp.asarray(cutoff_hz) * jnp.asarray(dt))
    norm = 1.0 / (1.0 + jnp.sqrt(2.0) * k + k * k)
    b0 = k * k * norm
    b1 = 2.0 * b0
    b2 = b0
    a1 = 2.0 * (k * k - 1.0) * norm
    a2 = (1.0 - jnp.sqrt(2.0) * k + k * k) * norm

    omega = 2.0 * jnp.pi * jnp.fft.rfftfreq(n)
    z1 = jnp.exp(-1j * omega)
    numerator = b0 + b1 * z1 + b2 * z1 * z1
    denominator = 1.0 + a1 * z1 + a2 * z1 * z1
    # A forward-backward filter has response H(z) H(z^-1) = |H(z)|^2.
    zero_phase_gain = jnp.real(numerator * jnp.conj(numerator)) / jnp.real(
        denominator * jnp.conj(denominator)
    )
    return jnp.fft.irfft(jnp.fft.rfft(values) * zero_phase_gain, n=n)


def delay_samples_from_seconds(delay_seconds, dt):
    return jnp.maximum(jnp.floor(delay_seconds / dt).astype(jnp.int32), 0)


def treadmill_controller_indices(n_nodes, delay_samples):
    node_idx = jnp.arange(n_nodes, dtype=jnp.int32)
    return {
        "next": (node_idx + 1) % n_nodes,
        "prev": (node_idx - 1) % n_nodes,
        "pre_prev": (node_idx - 2) % n_nodes,
        "delayed": (node_idx - delay_samples) % n_nodes,
        "delayed_pre": (node_idx - delay_samples - 1) % n_nodes,
    }


def treadmill_force_derivative_indices(n_nodes, delay_samples):
    """Indices for dF[k] = (F[k + 1] - F[k]) / dt at k = node - delay."""
    node_idx = jnp.arange(n_nodes, dtype=jnp.int32)
    delay = jnp.maximum(jnp.asarray(delay_samples, dtype=jnp.int32), 1)
    return {
        "forward": (node_idx - delay + 1) % n_nodes,
        "base": (node_idx - delay) % n_nodes,
    }


def treadmill_error_derivative_indices(n_nodes, error_delay_samples):
    """Indices for de[k] = (e[k + 1] - e[k]) / dt at k = node - delay - 1."""
    node_idx = jnp.arange(n_nodes, dtype=jnp.int32)
    delay = jnp.maximum(jnp.asarray(error_delay_samples, dtype=jnp.int32), 0)
    return {
        "forward": (node_idx - delay) % n_nodes,
        "base": (node_idx - delay - 1) % n_nodes,
    }


def swing_mask(fy, swing_force_threshold):
    return fy < swing_force_threshold


def controller_next_speed(
    fy,
    fx,
    speed,
    target_speed,
    dt,
    delay_samples,
    error_delay_samples,
    k_fy,
    k_fx,
    k_p,
    k_d,
    swing_force_threshold,
    heel_strike_index=0,
    post_hs_interpolation_samples=0,
):
    fy = jnp.asarray(fy)
    fx = jnp.asarray(fx)
    speed = jnp.asarray(speed)
    # For now the controller uses only the post-HS interpolation. Keep the
    # zero-phase Butterworth helper available for offline comparisons, but do
    # not filter the AP force supplied to the controller.
    fx = cubic_bridge_post_hs(fx, heel_strike_index, post_hs_interpolation_samples)
    idx = treadmill_controller_indices(speed.shape[0], delay_samples)
    force_idx = treadmill_force_derivative_indices(speed.shape[0], delay_samples)
    error_idx = treadmill_controller_indices(speed.shape[0], error_delay_samples)
    error_derivative_idx = treadmill_error_derivative_indices(speed.shape[0], error_delay_samples)
    pd_error_idx = error_idx["delayed"]

    in_swing = swing_mask(fy, swing_force_threshold)
    error = jnp.where(in_swing, 0.0, target_speed - speed)
    stance_next = (
        speed
        + k_p * error[pd_error_idx]
        + k_d * (error[error_derivative_idx["forward"]] - error[error_derivative_idx["base"]]) / dt
        + k_fy * (fy[force_idx["forward"]] - fy[force_idx["base"]]) / dt
        + k_fx * (fx[force_idx["forward"]] - fx[force_idx["base"]]) / dt
    )
    return jnp.where(in_swing, target_speed, stance_next)


def controller_residual_side(
    fy,
    fx,
    speed,
    target_speed,
    dt,
    delay_samples,
    error_delay_samples,
    k_fy,
    k_fx,
    k_p,
    k_d,
    swing_force_threshold,
    heel_strike_index=0,
    post_hs_interpolation_samples=0,
):
    fy = jnp.asarray(fy)
    fx = jnp.asarray(fx)
    speed = jnp.asarray(speed)
    if speed.shape[0] > 1:
        fy = fy[:-1]
        fx = fx[:-1]
        speed = speed[:-1]
    idx = treadmill_controller_indices(speed.shape[0], delay_samples)
    predicted_next = controller_next_speed(
        fy=fy,
        fx=fx,
        speed=speed,
        target_speed=target_speed,
        dt=dt,
        delay_samples=delay_samples,
        error_delay_samples=error_delay_samples,
        k_fy=k_fy,
        k_fx=k_fx,
        k_p=k_p,
        k_d=k_d,
        swing_force_threshold=swing_force_threshold,
        heel_strike_index=heel_strike_index,
        post_hs_interpolation_samples=post_hs_interpolation_samples,
    )
    return speed[idx["next"]] - predicted_next


def controller_residual_both_sides(
    fy_r,
    fx_r,
    fy_l,
    fx_l,
    speed_r,
    speed_l,
    target_speed,
    dt,
    delay_samples,
    error_delay_samples,
    k_fy,
    k_fx,
    k_p,
    k_d,
    swing_force_threshold,
    post_hs_interpolation_samples=0,
    heel_strike_index_r=0,
    heel_strike_index_l=None,
    post_hs_interpolation_samples_r=None,
    post_hs_interpolation_samples_l=None,
):
    if heel_strike_index_l is None:
        heel_strike_index_l = speed_l.shape[0] // 2
    if post_hs_interpolation_samples_r is None:
        post_hs_interpolation_samples_r = post_hs_interpolation_samples
    if post_hs_interpolation_samples_l is None:
        post_hs_interpolation_samples_l = post_hs_interpolation_samples
    c_left = controller_residual_side(
        fy=fy_l,
        fx=fx_l,
        speed=speed_l,
        target_speed=target_speed,
        dt=dt,
        delay_samples=delay_samples,
        error_delay_samples=error_delay_samples,
        k_fy=k_fy,
        k_fx=k_fx,
        k_p=k_p,
        k_d=k_d,
        swing_force_threshold=swing_force_threshold,
        heel_strike_index=heel_strike_index_l,
        post_hs_interpolation_samples=post_hs_interpolation_samples_l,
    )
    c_right = controller_residual_side(
        fy=fy_r,
        fx=fx_r,
        speed=speed_r,
        target_speed=target_speed,
        dt=dt,
        delay_samples=delay_samples,
        error_delay_samples=error_delay_samples,
        k_fy=k_fy,
        k_fx=k_fx,
        k_p=k_p,
        k_d=k_d,
        swing_force_threshold=swing_force_threshold,
        heel_strike_index=heel_strike_index_r,
        post_hs_interpolation_samples=post_hs_interpolation_samples_r,
    )
    return jnp.concatenate((c_left, c_right), axis=0)
