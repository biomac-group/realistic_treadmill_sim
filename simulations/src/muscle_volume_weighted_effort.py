"""Muscle-volume-weighted effort adapted from Nitschke et al. (2020), Eq. 9."""

from __future__ import annotations

import importlib
import os
from functools import partial

import jax
import jax.numpy as jnp
from biosym.objectives.base_objective import BaseObjective


class Objective(BaseObjective):
    """Volume-weighted powered activation normalized by muscle count and speed."""

    def __init__(self, model, settings, **kwargs):
        actuators = model.actuators
        constants = getattr(actuators, "muscle_constants", None)
        if constants is None or "fmax" not in constants or "lceopt" not in constants:
            raise TypeError("This objective requires Hill2d fmax and lceopt constants.")
        volume_proxy = (
            jnp.asarray(constants["fmax"]).reshape(-1)
            * jnp.asarray(constants["lceopt"]).reshape(-1)
        )
        self.volume_fractions = volume_proxy / jnp.sum(volume_proxy)
        self.activation_indices = tuple(int(index) for index in actuators.idx["a"])
        self.n_nodes = int(settings["nnodes"])
        self.n_muscles = len(self.activation_indices)
        self.exponent = float(kwargs.get("exponent", 3))
        self.reference_speed = abs(float(kwargs.get("reference_speed", 0.0)))
        self.speed_normalization = bool(kwargs.get("speed_normalization", True))
        if self.speed_normalization and self.reference_speed <= 0.0:
            raise ValueError("reference_speed must be positive for speed normalization.")

    def _get_info(self):
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Fmax*lceopt weighted powered activation effort.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "activation_indices": self.activation_indices,
            "n_nodes": self.n_nodes,
            "n_muscles": self.n_muscles,
            "exponent": self.exponent,
            "reference_speed": self.reference_speed,
            "speed_normalization": self.speed_normalization,
            "signal": "activation (Nitschke et al. use neural excitation)",
        }

    def get_objfun(self):
        return jax.jit(partial(objfun, volume_fractions=self.volume_fractions, info=self._get_info()))

    def get_gradient(self):
        fun = partial(objfun, volume_fractions=self.volume_fractions, info=self._get_info())
        return jax.jit(jax.grad(fun, argnums=(0, 1)))


def objfun(states, globals_dict, *, volume_fractions, info):
    del globals_dict
    activation = states.actuator_model[: info["n_nodes"], info["activation_indices"]]
    per_node = jnp.sum(
        volume_fractions * jnp.abs(activation) ** info["exponent"], axis=1
    )
    effort = jnp.mean(per_node) / info["n_muscles"]
    if info["speed_normalization"]:
        effort = effort / info["reference_speed"] ** info["exponent"]
    return effort


def install_as_biosym_objective():
    biosym_objfun = importlib.import_module("biosym.ocp.objfun")
    biosym_objfun.muscle_volume_weighted_effort = importlib.import_module(__name__)


def select_effort_objective(
    collocation_settings,
    effort_model,
    speed,
    weight=None,
    speedweighting=False,
):
    """Configure BioSym's effort term using the prescribed speed as a constant.

    BioSym's built-in ``speedweighting`` reads ``globals.speed``. That is the
    prescribed forward speed for overground gait, but it represents mean pelvis
    translation in the treadmill problem. Apply the mathematically equivalent
    constant factor to the objective weight instead, so both conditions use the
    requested walking/belt speed supplied by the pipeline.
    """
    if effort_model not in {"activation", "muscle_volume"}:
        raise ValueError(f"Unknown effort model: {effort_model}")
    for objective in collocation_settings.get("objectives", []):
        if objective.get("name") == "effort_term":
            args = dict(objective.get("args", {}))
            exponent = float(args.get("exponent", 2))
            args["weighting"] = (
                "volumeweighted" if effort_model == "muscle_volume" else "equal"
            )
            # Never let BioSym divide by the optimized globals.speed here.
            args["speedweighting"] = False
            objective["args"] = args
            base_weight = float(weight if weight is not None else objective["weight"])
            if speedweighting:
                prescribed_speed = abs(float(speed))
                if prescribed_speed <= 0.0:
                    raise ValueError(
                        "The prescribed speed must be positive for effort speed weighting."
                    )
                base_weight /= prescribed_speed**exponent
            objective["weight"] = base_weight
            return
    raise ValueError("The tracking YAML has no effort_term objective to replace.")
