"""Shared helpers for metabolic-rate post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Formulation = Literal["continuous", "original"]


@dataclass(frozen=True)
class MetabolicRateResult:
    """Container returned by metabolic-rate post-processing functions."""

    by_muscle: np.ndarray
    total: np.ndarray
    average_total: float | None
    average_per_muscle: np.ndarray
    metabolic_rate_w_per_kg: float | None
    metabolic_cost_j_per_kg_m: float | None
    cost_of_transport: float | None
    metabolic_cost_per_muscle_j_per_kg_m: np.ndarray | None
    muscle_names: list[str]


def as_flat_array(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def as_nodes_by_muscles(value, n_muscles: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.shape[0] == n_muscles:
        return array.T
    return array


def positive_part_smooth(x: np.ndarray, epsilon: float) -> np.ndarray:
    return 0.5 * (x + np.sqrt(x * x + epsilon * epsilon))


def get_body_mass(biosym_model) -> float:
    if hasattr(biosym_model, "variables"):
        variables = biosym_model.variables
        mask = (variables["name"].str.startswith("m_")) & (variables["type"] == "constant")
        return float(variables.loc[mask, "x0"].sum())

    constants = biosym_model.default_inputs.constants.model
    return float(
        sum(
            constants[i]
            for i, name in enumerate(biosym_model.constants)
            if name.startswith("m_")
        )
    )


def summarize_metabolic_rate(
    biosym_model,
    by_muscle: np.ndarray,
    muscle_names: list[str],
    *,
    duration: float | None,
    body_mass: float | None,
    speed: float | None,
    gravity: float,
    match_toolbox_node_average: bool,
    include_resting_rate: bool,
) -> MetabolicRateResult:
    total = by_muscle.sum(axis=1)
    average_nodes = by_muscle[:-1] if match_toolbox_node_average and by_muscle.shape[0] > 1 else by_muscle
    average_per_muscle = average_nodes.mean(axis=0)
    average_total = float(average_per_muscle.sum())
    if duration is not None:
        time = np.linspace(0.0, float(duration), total.size)
        average_total = float(np.trapezoid(total, time) / float(duration))

    metabolic_rate_w_per_kg = None
    metabolic_cost_j_per_kg_m = None
    cost_of_transport = None
    metabolic_cost_per_muscle = None
    if body_mass is None:
        body_mass = get_body_mass(biosym_model)
    if body_mass:
        metabolic_rate_w_per_kg = average_total / float(body_mass)
        if speed is not None:
            speed = abs(float(speed))
            if speed > 0.0:
                offset = 1.0 if include_resting_rate else 0.0
                metabolic_cost_j_per_kg_m = (metabolic_rate_w_per_kg + offset) / speed
                cost_of_transport = metabolic_cost_j_per_kg_m / float(gravity)
                metabolic_cost_per_muscle = average_per_muscle / float(body_mass) / speed

    return MetabolicRateResult(
        by_muscle=by_muscle,
        total=total,
        average_total=average_total,
        average_per_muscle=average_per_muscle,
        metabolic_rate_w_per_kg=metabolic_rate_w_per_kg,
        metabolic_cost_j_per_kg_m=metabolic_cost_j_per_kg_m,
        cost_of_transport=cost_of_transport,
        metabolic_cost_per_muscle_j_per_kg_m=metabolic_cost_per_muscle,
        muscle_names=muscle_names,
    )
