"""Track GRFs calculated directly by Biosym's state-free contact model."""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.objectives.base_objective import BaseObjective
from biosym.utils import read_mot


class Objective(BaseObjective):
    def __init__(self, model, settings, **kwargs):
        self.model = model
        self.n_nodes = int(settings["nnodes"])
        grf = read_mot(kwargs["presegmented_file"])
        self.grf_exp = jnp.asarray(grf.filter(like="_mean").values)
        self.grf_var = jnp.asarray(grf.filter(like="_var").values) + 1e-8
        self.body_weight = jnp.sum(model.default_constants.mass) * 9.81
        threshold = kwargs.get("force_threshold")
        self.force_threshold = None if threshold is None else float(threshold)
        if self.grf_exp.shape != (self.n_nodes, model.ext_forces.n):
            raise ValueError(
                f"GRF reference has shape {self.grf_exp.shape}; expected "
                f"({self.n_nodes}, {model.ext_forces.n})."
            )

    def _get_info(self):
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Track GRFs evaluated from Hunt-Crossley contact.",
            "required_variables": {"states": ["model"], "constants": ["model"]},
            "force_threshold": self.force_threshold,
        }

    def get_objfun(self):
        return jax.jit(
            partial(
                objfun,
                model=self.model,
                constants=self.model.default_constants,
                grf_exp=self.grf_exp,
                grf_var=self.grf_var,
                body_weight=self.body_weight,
                force_threshold=self.force_threshold,
            )
        )

    def get_gradient(self):
        fun = partial(
            objfun,
            model=self.model,
            constants=self.model.default_constants,
            grf_exp=self.grf_exp,
            grf_var=self.grf_var,
            body_weight=self.body_weight,
            force_threshold=self.force_threshold,
        )
        return jax.jit(jax.grad(fun, argnums=(0, 1)))


def _materialize(state, model):
    return state.replace(
        qdd=jnp.zeros(model.accs.n),
        tau=jnp.zeros(model.tau.n),
        ext_forces=jnp.zeros(model.ext_forces.n),
        ext_torques=jnp.zeros(model.ext_torques.n),
    )


def objfun(
    states,
    globals_dict,
    *,
    model,
    constants,
    grf_exp,
    grf_var,
    body_weight,
    force_threshold,
):
    def contact_force(state):
        forces, _ = model.run["gc_model"](_materialize(state, model), constants)
        # Model order is right then left; experimental MOT columns are left then right.
        return jnp.concatenate((forces[1], forces[0]))

    simulated = jax.vmap(contact_force)(states[: grf_exp.shape[0]])
    simulated_bw = simulated / body_weight
    experimental_bw = grf_exp / body_weight
    # Apply one elementwise mask to every experimental GRF component. Each AP
    # or vertical value is tracked only when that same value is above +5 N or
    # below -5 N; no vertical-force stance mask is propagated to AP force.
    mask = (
        jnp.ones_like(grf_exp, dtype=bool)
        if force_threshold is None
        else jnp.abs(grf_exp) > force_threshold
    )
    weighted_error = (simulated_bw - experimental_bw) ** 2 / grf_var
    return jnp.sum(jnp.where(mask, weighted_error, 0.0)) / jnp.maximum(
        jnp.sum(mask), 1
    )
