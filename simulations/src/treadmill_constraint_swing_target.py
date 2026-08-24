from biosym.utils.states import *


import os
from functools import partial

import jax
import jax.numpy as jnp

from biosym.constraints.base_constraint import BaseConstraint
from biosym.ocp import utils as ocp_utils
from src.treadmill_controller import controller_residual_both_sides


class Constraint(BaseConstraint):
    """
    Treadmill speed constraint with swing-phase target-speed enforcement.

    During stance, the treadmill belt follows the shared delayed PD / GRF-based
    controller. The delayed GRF terms and PD error derivative use forward
    differences through src.treadmill_controller. When vertical GRF is below
    the swing threshold, the belt speed is driven directly to the desired
    treadmill speed.
    """

    def __init__(self, model, settings, args):
        self.model = model
        self.settings = settings.copy()
        self.args = args or {}
        self.settings["nvpn"] = model.opt_states.size()
        self.nvar = settings.get("nvar")
        self.nnodes_dur = self.settings.get("nnodes_dur")
        self.delay = int(self.args.get("delay", 0))
        self.error_delay_samples = int(self.args.get("error_delay_samples", 1))
        self.post_hs_interpolation_samples = int(
            self.args.get("post_hs_interpolation_samples", 0)
        )
        self.heel_strike_index_r = int(self.args.get("heel_strike_index_r", 0))
        self.heel_strike_index_l = int(
            self.args.get("heel_strike_index_l", self.nnodes_dur // 2)
        )
        self.post_hs_interpolation_samples_r = int(
            self.args.get("post_hs_interpolation_samples_r", self.post_hs_interpolation_samples)
        )
        self.post_hs_interpolation_samples_l = int(
            self.args.get("post_hs_interpolation_samples_l", self.post_hs_interpolation_samples)
        )

        self.k_fy = float(self.args.get("k_fy", 0.0))
        self.k_fx = float(self.args.get("k_fx", 0.0))
        self.k_p = float(self.args.get("k_p", 0.0))
        self.k_d = float(self.args.get("k_d", 0.0))
        self.target_speed = self.args.get("target_speed")
        self.use_global_speed = self.target_speed is None
        self.swing_force_threshold = float(self.args.get("swing_force_threshold", 10.0))

        self.fy_r_idx = self.model.state_vector.index("f_foot_r_y")
        self.fx_r_idx = self.model.state_vector.index("f_foot_r_x")
        self.fy_l_idx = self.model.state_vector.index("f_foot_l_y")
        self.fx_l_idx = self.model.state_vector.index("f_foot_l_x")
        self.gc_offset = len(self.model.state_vector)
        self.speed_r_idx = self.gc_offset
        self.speed_l_idx = self.gc_offset + 1
        self.nnodes_dur = self.settings.get("nnodes_dur")

    def _get_info(self):
        return {
            "name": os.path.splitext(os.path.basename(__file__))[0],
            "description": "Treadmill speed controller with forward-difference GRF/PD derivative terms and swing-phase target speed enforcement.",
            "required_variables": {"states": ["model", "gc_model"]},
            "nnz": self.get_nnz(),
            "ncons": self.get_n_constraints(),
            "fy_r_idx": self.fy_r_idx,
            "fx_r_idx": self.fx_r_idx,
            "fy_l_idx": self.fy_l_idx,
            "fx_l_idx": self.fx_l_idx,
            "speed_r_idx": self.speed_r_idx,
            "speed_l_idx": self.speed_l_idx,
            "use_global_speed": self.use_global_speed,
            "nnodes_dur": self.nnodes_dur,
            "post_hs_interpolation_samples": self.post_hs_interpolation_samples,
            "heel_strike_index_r": self.heel_strike_index_r,
            "heel_strike_index_l": self.heel_strike_index_l,
            "post_hs_interpolation_samples_r": self.post_hs_interpolation_samples_r,
            "post_hs_interpolation_samples_l": self.post_hs_interpolation_samples_l,
        }

    def get_confun(self):
        return jax.jit(
            partial(
                confun,
                settings=self.settings,
                info=self._get_info(),
                delay=self.delay,
                error_delay_samples=self.error_delay_samples,
                k_fy=self.k_fy,
                k_fx=self.k_fx,
                k_p=self.k_p,
                k_d=self.k_d,
                target_speed=self.target_speed,
                swing_force_threshold=self.swing_force_threshold,
                model=self.model,
            )
        )

    def get_jacobian(self):
        return jax.jit(
            partial(
                jacobian,
                settings=self.settings,
                info=self._get_info(),
                delay=self.delay,
                error_delay_samples=self.error_delay_samples,
                k_fy=self.k_fy,
                k_fx=self.k_fx,
                k_p=self.k_p,
                k_d=self.k_d,
                target_speed=self.target_speed,
                swing_force_threshold=self.swing_force_threshold,
                model=self.model,
            )
        )

    def get_n_constraints(self):
        return 2 * (self.nnodes_dur - 1)

    def get_nnz(self):
        total_cols = self.nnodes_dur * (self.settings["nvpn"]) + 2
        return self.get_n_constraints() * total_cols


def confun(
    states_list,
    globals_dict,
    settings,
    info,
    delay,
    error_delay_samples,
    k_fy,
    k_fx,
    k_p,
    k_d,
    target_speed,
    swing_force_threshold,
    model,
):
    if hasattr(states_list, "q"):
        n_nodes = settings.get("nnodes_dur")
        local_states = states_list[:n_nodes]
        constants = model.default_constants
        def contact_force(state):
            # Collocation only stores the optimized fields (q, qd, gc_model).
            # The model's symbolic contact function still expects every model
            # state, even though contact itself does not depend on these zeros.
            full_state = state.replace(
                qdd=jnp.zeros(model.accs.n),
                tau=jnp.zeros(model.tau.n),
                ext_forces=jnp.zeros(model.ext_forces.n),
                ext_torques=jnp.zeros(model.ext_torques.n),
            )
            return model.run["gc_model"](full_state, constants)[0]

        forces = jax.vmap(contact_force)(local_states)
        gc_states = local_states.gc_model
        return _predict_treadmill_speed_from_vectors(
            fy_r=forces[:, 0, 1],
            fx_r=forces[:, 0, 0],
            fy_l=forces[:, 1, 1],
            fx_l=forces[:, 1, 0],
            speed_r=gc_states[:, 0],
            speed_l=gc_states[:, 1],
            delay=delay,
            error_delay_samples=error_delay_samples,
            k_fy=k_fy,
            k_fx=k_fx,
            k_p=k_p,
            k_d=k_d,
            target_speed=_get_target_speed(globals_dict, info, target_speed),
            swing_force_threshold=swing_force_threshold,
            post_hs_interpolation_samples=info.get("post_hs_interpolation_samples", 0),
            heel_strike_index_r=info.get("heel_strike_index_r", 0),
            heel_strike_index_l=info.get("heel_strike_index_l"),
            post_hs_interpolation_samples_r=info.get("post_hs_interpolation_samples_r"),
            post_hs_interpolation_samples_l=info.get("post_hs_interpolation_samples_l"),
            dt=globals_dict.dur / settings.get("nnodes"),
        )
    model_states = states_list.states.model[: settings.get("nnodes_dur")]
    gc_states = states_list.states.gc_model[: settings.get("nnodes_dur")]
    return _predict_treadmill_speed_from_states(
        model_states,
        gc_states,
        info,
        delay=delay,
        error_delay_samples=error_delay_samples,
        k_fy=k_fy,
        k_fx=k_fx,
        k_p=k_p,
        k_d=k_d,
        target_speed=_get_target_speed(globals_dict, info, target_speed),
        swing_force_threshold=swing_force_threshold,
        dt=globals_dict.dur / (settings.get("nnodes")),
    )


def jacobian(
    states_list,
    globals_dict,
    settings,
    info,
    delay,
    error_delay_samples,
    k_fy,
    k_fx,
    k_p,
    k_d,
    target_speed,
    swing_force_threshold,
    model,
):
    template_globals = globals_dict
    x = ocp_utils.states_dict_to_x(states_list, template_globals)

    def flat_confun(flat_x):
        local_states, local_globals = ocp_utils.x_to_states_dict(flat_x, states_list, template_globals)
        return confun(
            local_states,
            local_globals,
            settings=settings,
            info=info,
            delay=delay,
            error_delay_samples=error_delay_samples,
            k_fy=k_fy,
            k_fx=k_fx,
            k_p=k_p,
            k_d=k_d,
            target_speed=target_speed,
            swing_force_threshold=swing_force_threshold,
            model=model,
        )

    dense = jax.jacobian(flat_confun)(x)
    n_constraints = dense.shape[0]
    ncols = dense.shape[1]
    rows = jnp.repeat(jnp.arange(n_constraints, dtype=jnp.int32), ncols)
    cols = jnp.tile(jnp.arange(ncols, dtype=jnp.int32), n_constraints)
    data = dense.reshape(-1)
    return rows, cols, data


def _get_target_speed(globals_dict, info, target_speed):
    if info["use_global_speed"]:
        return jnp.asarray(globals_dict.speed).reshape(())
    return jnp.asarray(target_speed).reshape(())


def _predict_treadmill_speed_from_states(
    model_states,
    gc_states,
    info,
    delay,
    error_delay_samples,
    k_fy,
    k_fx,
    k_p,
    k_d,
    target_speed,
    swing_force_threshold,
    dt,
):
    return _predict_treadmill_speed_from_vectors(
        fy_r=model_states[:, info["fy_r_idx"]],
        fx_r=model_states[:, info["fx_r_idx"]],
        fy_l=model_states[:, info["fy_l_idx"]],
        fx_l=model_states[:, info["fx_l_idx"]],
        speed_r=gc_states[:, 0],
        speed_l=gc_states[:, 1],
        delay=delay,
        error_delay_samples=error_delay_samples,
        k_fy=k_fy,
        k_fx=k_fx,
        k_p=k_p,
        k_d=k_d,
        target_speed=target_speed,
        swing_force_threshold=swing_force_threshold,
        post_hs_interpolation_samples=info.get("post_hs_interpolation_samples", 0),
        heel_strike_index_r=info.get("heel_strike_index_r", 0),
        heel_strike_index_l=info.get("heel_strike_index_l"),
        post_hs_interpolation_samples_r=info.get("post_hs_interpolation_samples_r"),
        post_hs_interpolation_samples_l=info.get("post_hs_interpolation_samples_l"),
        dt=dt,
    )


def _predict_treadmill_speed_from_vectors(
    fy_r,
    fx_r,
    fy_l,
    fx_l,
    speed_r,
    speed_l,
    delay,
    error_delay_samples,
    k_fy,
    k_fx,
    k_p,
    k_d,
    target_speed,
    swing_force_threshold,
    post_hs_interpolation_samples,
    heel_strike_index_r,
    heel_strike_index_l,
    post_hs_interpolation_samples_r,
    post_hs_interpolation_samples_l,
    dt,
):
    return controller_residual_both_sides(
        fy_r=fy_r,
        fx_r=fx_r,
        fy_l=fy_l,
        fx_l=fx_l,
        speed_r=speed_r,
        speed_l=speed_l,
        target_speed=target_speed,
        dt=dt,
        delay_samples=delay,
        error_delay_samples=error_delay_samples,
        k_fy=k_fy,
        k_fx=k_fx,
        k_p=k_p,
        k_d=k_d,
        swing_force_threshold=swing_force_threshold,
        post_hs_interpolation_samples=post_hs_interpolation_samples,
        heel_strike_index_r=heel_strike_index_r,
        heel_strike_index_l=heel_strike_index_l,
        post_hs_interpolation_samples_r=post_hs_interpolation_samples_r,
        post_hs_interpolation_samples_l=post_hs_interpolation_samples_l,
    )
