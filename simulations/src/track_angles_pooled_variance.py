"""Angle tracking with one variance normalization shared across speeds."""

from __future__ import annotations

import importlib
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from biosym.objectives.base_objective import BaseObjective
from biosym.utils import read_mot


def _periodic_resample(values, n_nodes):
    values = np.asarray(values, dtype=float)
    if values.shape[0] == n_nodes:
        return values
    old_phase = np.arange(values.shape[0] + 1) / values.shape[0]
    new_phase = np.arange(n_nodes) / n_nodes
    closed = np.vstack((values, values[:1]))
    return np.column_stack([
        np.interp(new_phase, old_phase, closed[:, column])
        for column in range(values.shape[1])
    ])


class Objective(BaseObjective):
    def __init__(self, model, settings, **kwargs):
        self.n_nodes = int(settings["nnodes"])
        names = list(model.coordinates.names)
        excluded = set(kwargs.get("exclude") or [])
        self.tracked_indices = tuple(
            index for index, name in enumerate(names)
            if name.removeprefix("q_") not in excluded
        )
        reference = read_mot(kwargs["presegmented_file"])
        mean = _periodic_resample(reference.filter(like="_mean").values, self.n_nodes)
        variance_files = kwargs.get("pooled_variance_files")
        if not variance_files:
            raise ValueError("No across-speed angle variance files were provided.")
        variances = [
            _periodic_resample(read_mot(path).filter(like="_var").values, self.n_nodes)
            for path in variance_files
        ]
        pooled = np.mean(np.stack(variances), axis=0)
        columns = list(self.tracked_indices)
        self.obj_settings = {
            "q_exp": jnp.asarray(mean[:, columns]),
            "q_var": jnp.asarray(pooled[:, columns]) + 1e-8,
        }
        self.variance_files = tuple(str(path) for path in variance_files)

    def _get_info(self):
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Angle tracking with variance pooled across speeds.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "n_nodes": self.n_nodes,
            "tracked_indices": self.tracked_indices,
            "pooled_variance_files": self.variance_files,
        }

    def get_objfun(self):
        return jax.jit(partial(objfun, settings=self.obj_settings, info=self._get_info()))

    def get_gradient(self):
        fun = partial(objfun, settings=self.obj_settings, info=self._get_info())
        return jax.jit(jax.grad(fun, argnums=(0, 1)))


def objfun(states, globals_dict, *, settings, info):
    del globals_dict
    simulated = states.q[: info["n_nodes"], info["tracked_indices"]]
    return jnp.mean((simulated - settings["q_exp"]) ** 2 / settings["q_var"])


def install_as_biosym_track_angles():
    biosym_objfun = importlib.import_module("biosym.ocp.objfun")
    biosym_objfun.track_angles = importlib.import_module(__name__)
