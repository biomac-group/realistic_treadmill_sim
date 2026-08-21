"""Biosym Hunt-Crossley sphere/point contact against two moving treadmill belts."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from sympy import Matrix, lambdify

from biosym.model.contact.multi_contact import MultiContact

from src.contact_huntcrossley import (
    FRICTION_DIRECTION_EPS,
    FRICTION_MODEL_B_NORMAL_FORCE_THRESHOLD_BW,
    FRICTION_MODEL_B_NORMAL_FORCE_TRANSITION_BW,
    FRICTION_LOADING_RATE_TRANSITION_BW_PER_S,
    HuntCrossley,
    directional_friction_ramp,
    heel_friction_activation,
    kinematic_time_derivative,
)


class HuntCrossleyTreadmill(HuntCrossley):
    """Hunt-Crossley contact with one fore-aft belt-speed state per foot.

    Geometry, penetration, normal velocity, and Hunt-Crossley parameters are
    identical to the stationary model. Only tangential relative velocity is
    changed from ``v_foot`` to ``v_foot - v_belt``.
    """

    def __init__(self, stationary_contact: HuntCrossley):
        super().__init__(
            stationary_contact.pairs,
            stiffness=stationary_contact.k,
            dissipation=stationary_contact.c,
            static_friction=stationary_contact.us,
            dynamic_friction=stationary_contact.ud,
            viscous_friction=stationary_contact.uv,
            transition_velocity=stationary_contact.v_t,
            eps_depth=stationary_contact.eps_depth,
            eps_diss=stationary_contact.eps_diss,
            grad_bias=stationary_contact.grad_bias,
            friction_formulation=stationary_contact.friction_formulation,
            loading_rate_threshold_bw_per_s=stationary_contact.loading_rate_threshold_bw_per_s,
            heel_force_average_dt=stationary_contact.heel_force_average_dt,
        )
        self.state_vector = ["treadmill_speed_r", "treadmill_speed_l"]

    def get_n_states(self):
        return 2

    def get_states(self):
        return list(self.state_vector)

    def process_eom(self, model, **kwargs):
        body_weight = kwargs.get("body_weight")
        if body_weight is None:
            mass = float(np.asarray(model.default_constants.mass).sum())
            gravity = float(np.abs(np.asarray(model.default_constants.g)).max())
            body_weight = mass * gravity
        self.model_b_normal_force_threshold = (
            FRICTION_MODEL_B_NORMAL_FORCE_THRESHOLD_BW * float(body_weight)
        )
        self.model_b_normal_force_transition = (
            FRICTION_MODEL_B_NORMAL_FORCE_TRANSITION_BW * float(body_weight)
        )
        self.body_weight = float(body_weight)
        self.loading_rate_threshold = self.loading_rate_threshold_bw_per_s * self.body_weight
        self.loading_rate_transition = FRICTION_LOADING_RATE_TRANSITION_BW_PER_S * self.body_weight
        self._build_geometries(model)
        body_keys = list(model.rigid_bodies.keys())
        depths, normal_velocities, normals = [], [], []
        tangential_velocities, tangential_accelerations, positions = [], [], []
        body_slots, fk_indices, belt_slots = [], [], []
        pair_normal_force_exprs, pair_bodies, is_heel = [], [], []

        for feature, other in self.pairs:
            penetration = feature.penetration(other)
            pair_normal_force_exprs.append(
                self._pair_normal_force(
                    penetration, self.hertz_coefficients[len(pair_normal_force_exprs)]
                )
            )
            pair_bodies.append(feature.parent_body)
            is_heel.append("heel" in feature.name.lower())
            depths.append(penetration["depth"])
            normal_velocities.append(penetration["normal_velocity"])
            normals.append(penetration["normal"].T)
            tangential_velocities.append(penetration["tangential_velocity"].T)
            tangential_accelerations.append(
                kinematic_time_derivative(
                    model, penetration["tangential_velocity"]
                ).T
            )
            positions.append(feature.pos_expr.T)
            body_slots.append(self._body_index(feature.parent_body))
            fk_indices.append(body_keys.index(feature.parent_body))
            if feature.parent_body.endswith("_r"):
                belt_slots.append(0)
            elif feature.parent_body.endswith("_l"):
                belt_slots.append(1)
            else:
                raise ValueError(
                    f"Cannot assign {feature.name!r} on {feature.parent_body!r} to a treadmill belt."
                )

        args = model._symbols
        self.depth_fn = lambdify(args, Matrix(depths), modules="jax", cse=True, docstring_limit=2)
        self.normal_velocity_fn = lambdify(
            args, Matrix(normal_velocities), modules="jax", cse=True, docstring_limit=2
        )
        self.normal_fn = lambdify(args, Matrix.vstack(*normals), modules="jax", cse=True, docstring_limit=2)
        self.tangential_velocity_fn = lambdify(
            args, Matrix.vstack(*tangential_velocities), modules="jax", cse=True, docstring_limit=2
        )
        self.tangential_acceleration_fn = lambdify(
            args,
            Matrix.vstack(*tangential_accelerations),
            modules="jax",
            cse=True,
            docstring_limit=2,
        )
        self.position_fn = lambdify(args, Matrix.vstack(*positions), modules="jax", cse=True, docstring_limit=2)
        foot_force_exprs = {}
        for force_expr, body in zip(pair_normal_force_exprs, pair_bodies):
            foot_force_exprs[body] = foot_force_exprs.get(body, 0) + force_expr
        pair_force_rate_exprs = [
            kinematic_time_derivative(model, Matrix([foot_force_exprs[body]]))[0]
            for body in pair_bodies
        ]
        self.foot_normal_force_rate_fn = lambdify(
            args, Matrix(pair_force_rate_exprs), modules="jax", cse=True, docstring_limit=2
        )
        pair_normal_force_rate_exprs = [
            kinematic_time_derivative(model, Matrix([force_expr]))[0]
            for force_expr in pair_normal_force_exprs
        ]
        self.pair_normal_force_rate_fn = lambdify(
            args,
            Matrix(pair_normal_force_rate_exprs),
            modules="jax",
            cse=True,
            docstring_limit=2,
        )
        self._body_slots = np.asarray(body_slots)
        self._fk_indices = np.asarray(fk_indices)
        self._belt_slots = np.asarray(belt_slots)
        self._is_heel = np.asarray(is_heel)
        # Names used by Biosym's inherited Hunt-Crossley plotter.
        self._contrib_body_slot = self._body_slots
        self._contrib_fk_idx = self._fk_indices

    def _contact_positions_and_forces(self, states, constants):
        s_flat = states.filter("model").flatten()
        c_flat = constants.filter("model").flatten()
        depth = jnp.asarray(self.depth_fn(*s_flat, *c_flat)).reshape(-1)
        normal_velocity = jnp.asarray(
            self.normal_velocity_fn(*s_flat, *c_flat)
        ).reshape(-1)
        normal = jnp.asarray(self.normal_fn(*s_flat, *c_flat)).reshape(-1, 3)
        tangential_velocity = jnp.asarray(
            self.tangential_velocity_fn(*s_flat, *c_flat)
        ).reshape(-1, 3)
        tangential_acceleration = jnp.asarray(
            self.tangential_acceleration_fn(*s_flat, *c_flat)
        ).reshape(-1, 3)
        positions = jnp.asarray(self.position_fn(*s_flat, *c_flat)).reshape(-1, 3)

        belt_speeds = jnp.asarray(states.gc_model).reshape(2)
        belt_velocity = jnp.zeros_like(tangential_velocity).at[:, 0].set(
            belt_speeds[self._belt_slots]
        )
        # Remove the belt's normal component as well, making this valid for a
        # tilted half-space even though the current treadmill is horizontal.
        belt_normal_speed = jnp.sum(belt_velocity * normal, axis=1)
        normal_velocity = normal_velocity + belt_normal_speed
        tangential_velocity = (
            tangential_velocity
            - belt_velocity
            + belt_normal_speed[:, None] * normal
        )

        x_pos = 0.5 * (jnp.sqrt(depth**2 + self.eps_depth**2) + depth)
        diss = 1.0 + 1.5 * self.c * normal_velocity
        diss_pos = 0.5 * (jnp.sqrt(diss**2 + self.eps_diss**2) + diss)
        coefficients = jnp.asarray(self.hertz_coefficients)
        f_n = coefficients * x_pos**1.5 * diss_pos + self.grad_bias * depth
        foot_normal_force = jnp.zeros(2).at[self._belt_slots].add(f_n)
        pair_foot_normal_force = foot_normal_force[self._belt_slots]
        pair_foot_normal_force_rate = jnp.asarray(
            self.foot_normal_force_rate_fn(*s_flat, *c_flat)
        ).reshape(-1)
        pair_normal_force_rate = jnp.asarray(
            self.pair_normal_force_rate_fn(*s_flat, *c_flat)
        ).reshape(-1)
        force_normal = f_n[:, None] * normal

        speed_sq = jnp.sum(tangential_velocity**2, axis=1)
        blend = self.ud + 2.0 * (self.us - self.ud) / (1.0 + speed_sq / self.v_t**2)
        speed = jnp.sqrt(speed_sq + FRICTION_DIRECTION_EPS**2)
        if self.friction_formulation in {"biosym", "biosym_heel_force_average", "biosym_force_activation"}:
            ramp = speed / jnp.sqrt(speed_sq + self.v_t**2)
        else:
            ramp = directional_friction_ramp(
                speed,
                tangential_acceleration[:, 0],
                self.v_t,
                pair_foot_normal_force,
                self.model_b_normal_force_threshold,
                self.model_b_normal_force_transition,
                pair_foot_normal_force_rate,
                self.loading_rate_threshold,
                self.loading_rate_transition,
                self.friction_formulation,
                f_n,
                self.body_weight,
                jnp.asarray(self._is_heel),
            )
        coulomb = (
            blend[:, None]
            * ramp[:, None]
            * tangential_velocity
            / speed[:, None]
        )
        friction_normal_force = f_n
        if self.friction_formulation == "biosym_heel_force_average":
            estimated_average = f_n - 0.5 * self.heel_force_average_dt * pair_normal_force_rate
            smoothing = 1e-6 * self.body_weight
            positive_average = 0.5 * (
                jnp.sqrt(estimated_average**2 + smoothing**2) + estimated_average
            )
            friction_normal_force = jnp.where(
                jnp.asarray(self._is_heel), positive_average, f_n
            )
        elif self.friction_formulation == "biosym_force_activation":
            activated_force = f_n * heel_friction_activation(
                pair_foot_normal_force, self.body_weight
            )
            friction_normal_force = jnp.where(
                jnp.asarray(self._is_heel), activated_force, f_n
            )
        force_friction = -friction_normal_force[:, None] * (
            coulomb + self.uv * tangential_velocity
        )
        forces = force_normal + force_friction
        return positions, forces

    def forward(self, states, constants, model):
        positions, forces = self._contact_positions_and_forces(states, constants)

        body_positions = model.run["FK"](states, constants)[self._fk_indices]
        moments = jnp.cross(positions - body_positions, forces)
        return (
            self._aggregate_to_bodies(forces, self._body_slots),
            self._aggregate_to_bodies(moments, self._body_slots),
        )

    def contact_slip_and_friction(self, states, constants):
        """Return per-sphere belt-relative slip, friction, and normal force."""
        s_flat = states.filter("model").flatten()
        c_flat = constants.filter("model").flatten()
        normal = jnp.asarray(self.normal_fn(*s_flat, *c_flat)).reshape(-1, 3)
        slip = jnp.asarray(
            self.tangential_velocity_fn(*s_flat, *c_flat)
        ).reshape(-1, 3)
        belt_speeds = jnp.asarray(states.gc_model).reshape(2)
        belt_velocity = jnp.zeros_like(slip).at[:, 0].set(
            belt_speeds[self._belt_slots]
        )
        belt_normal_speed = jnp.sum(belt_velocity * normal, axis=1)
        slip = slip - belt_velocity + belt_normal_speed[:, None] * normal
        _, force = self._contact_positions_and_forces(states, constants)
        normal_force = jnp.sum(force * normal, axis=1)
        friction = force - normal_force[:, None] * normal
        return slip, friction, normal_force

    def plot(self, states, model, mode, ax, **kwargs):
        """Use Biosym's standard plotter with moving-belt per-sphere forces."""
        if mode == "init":
            sequence = states.states if hasattr(states, "states") else states
            constants = (
                states.constants
                if hasattr(states, "constants")
                else model.default_constants
            )
            leading_shape = np.asarray(sequence.q).shape[:-1]
            sequence = sequence.replace(
                qdd=jnp.zeros((*leading_shape, model.accs.n)),
                tau=jnp.zeros((*leading_shape, model.tau.n)),
                ext_forces=jnp.zeros((*leading_shape, model.ext_forces.n)),
                ext_torques=jnp.zeros((*leading_shape, model.ext_torques.n)),
            )
            parent_states = sequence
        else:
            sequence = self._plot_states
            constants = self._plot_constants
            parent_states = states

        frame = kwargs.get("frame", 0)
        state = sequence[frame] if np.asarray(sequence.q).ndim > 1 else sequence
        positions, forces = self._contact_positions_and_forces(state, constants)

        # The parent plot method calls these with flattened model variables and
        # has no argument for gc_model belt-speed states. Supply the already
        # evaluated arrays for this frame so its drawing/update logic is reused.
        self.pos_fn = lambda *unused: np.asarray(positions)
        self.force_fn = lambda *unused: np.asarray(forces)
        return super().plot(parent_states, model, mode, ax, **kwargs)


class MultiHuntCrossleyTreadmill(HuntCrossleyTreadmill):
    """One treadmill model with per-contact parameters from several groups."""

    def __init__(self, stationary_contact: MultiContact):
        if not all(isinstance(contact, HuntCrossley) for contact in stationary_contact.models):
            names = [type(contact).__name__ for contact in stationary_contact.models]
            raise TypeError(f"All contact groups must be HuntCrossley; got {names}.")
        groups = stationary_contact.models
        first = groups[0]
        combined = HuntCrossley(
            [pair for contact in groups for pair in contact.pairs],
            stiffness=first.k,
            dissipation=first.c,
            static_friction=first.us,
            dynamic_friction=first.ud,
            viscous_friction=first.uv,
            transition_velocity=first.v_t,
            eps_depth=first.eps_depth,
            eps_diss=first.eps_diss,
            grad_bias=first.grad_bias,
        )
        super().__init__(combined)
        self.hertz_coefficients = np.concatenate(
            [contact.hertz_coefficients for contact in groups]
        )
        self._c_by_pair = np.asarray([contact.c for contact in groups for _ in contact.pairs])
        self._us_by_pair = np.asarray([contact.us for contact in groups for _ in contact.pairs])
        self._ud_by_pair = np.asarray([contact.ud for contact in groups for _ in contact.pairs])
        self._uv_by_pair = np.asarray([contact.uv for contact in groups for _ in contact.pairs])
        self._vt_by_pair = np.asarray([contact.v_t for contact in groups for _ in contact.pairs])

    def _contact_positions_and_forces(self, states, constants):
        s_flat = states.filter("model").flatten()
        c_flat = constants.filter("model").flatten()
        depth = jnp.asarray(self.depth_fn(*s_flat, *c_flat)).reshape(-1)
        normal_velocity = jnp.asarray(self.normal_velocity_fn(*s_flat, *c_flat)).reshape(-1)
        normal = jnp.asarray(self.normal_fn(*s_flat, *c_flat)).reshape(-1, 3)
        tangential_velocity = jnp.asarray(
            self.tangential_velocity_fn(*s_flat, *c_flat)
        ).reshape(-1, 3)
        tangential_acceleration = jnp.asarray(
            self.tangential_acceleration_fn(*s_flat, *c_flat)
        ).reshape(-1, 3)
        positions = jnp.asarray(self.position_fn(*s_flat, *c_flat)).reshape(-1, 3)

        belt_speeds = jnp.asarray(states.gc_model).reshape(2)
        belt_velocity = jnp.zeros_like(tangential_velocity).at[:, 0].set(
            belt_speeds[self._belt_slots]
        )
        belt_normal_speed = jnp.sum(belt_velocity * normal, axis=1)
        normal_velocity = normal_velocity + belt_normal_speed
        tangential_velocity = (
            tangential_velocity - belt_velocity + belt_normal_speed[:, None] * normal
        )

        k = jnp.asarray(self.hertz_coefficients)
        c = jnp.asarray(self._c_by_pair)
        us = jnp.asarray(self._us_by_pair)
        ud = jnp.asarray(self._ud_by_pair)
        uv = jnp.asarray(self._uv_by_pair)
        vt = jnp.asarray(self._vt_by_pair)
        x_pos = 0.5 * (jnp.sqrt(depth**2 + self.eps_depth**2) + depth)
        diss = 1.0 + 1.5 * c * normal_velocity
        diss_pos = 0.5 * (jnp.sqrt(diss**2 + self.eps_diss**2) + diss)
        f_n = k * x_pos**1.5 * diss_pos + self.grad_bias * depth
        foot_normal_force = jnp.zeros(2).at[self._belt_slots].add(f_n)
        pair_foot_normal_force = foot_normal_force[self._belt_slots]
        force_normal = f_n[:, None] * normal
        speed_sq = jnp.sum(tangential_velocity**2, axis=1)
        blend = ud + 2.0 * (us - ud) / (1.0 + speed_sq / vt**2)
        speed = jnp.sqrt(speed_sq + FRICTION_DIRECTION_EPS**2)
        ramp = directional_friction_ramp(
            speed,
            tangential_acceleration[:, 0],
            vt,
            pair_foot_normal_force,
            self.model_b_normal_force_threshold,
            self.model_b_normal_force_transition,
        )
        coulomb = (
            blend[:, None]
            * ramp[:, None]
            * tangential_velocity
            / speed[:, None]
        )
        forces = force_normal - f_n[:, None] * (
            coulomb + uv[:, None] * tangential_velocity
        )
        return positions, forces
