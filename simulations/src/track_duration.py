import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp

from biosym.objectives.base_objective import BaseObjective
from src.gait_data import DEFAULT_GAIT_DATA_FILE, get_duration


class Objective(BaseObjective):
    """
    Objective term for tracking the experimental gait-cycle duration.
    """

    def __init__(self, model, settings, **kwargs):
        self.model = model
        self.settings = settings

        datafile = Path(kwargs.get("datafile", DEFAULT_GAIT_DATA_FILE))
        participant = kwargs.get("participant", kwargs.get("person"))
        if participant is None:
            raise ValueError("track_duration objective requires 'person' or 'participant'.")
        participant = int(participant)
        speed = round(abs(float(kwargs["speed"])), 1)
        self.duration_exp = get_duration(datafile, participant, speed)
        self.obj_settings = {"duration_exp": jnp.asarray(self.duration_exp)}

    def _get_info(self):
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Objective term for tracking experimental trial duration.",
            "required_variables": {"globals": ["dur"]},
        }

    def get_objfun(self):
        fun = partial(objfun, settings=self.obj_settings, info=self._get_info())
        return jax.jit(fun)

    def get_gradient(self):
        fun = partial(objfun, settings=self.obj_settings, info=self._get_info())
        return jax.jit(jax.grad(fun, argnums=[0, 1]))


def objfun(states_list, globals_dict, settings, info):
    del states_list, info
    duration_error = globals_dict.dur - settings["duration_exp"]
    return duration_error ** 2
