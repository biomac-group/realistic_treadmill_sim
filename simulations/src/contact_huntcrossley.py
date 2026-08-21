"""Repository-local OpenSim-compatible Hunt--Crossley contact law.

BioSym 0.1.8 treats the OpenSim ``stiffness`` property as the coefficient
that directly multiplies indentation**(3/2).  OpenSim/Simbody instead treats
it as the plane-strain modulus of each contact material.  The composite
modulus and the radius-dependent Hertz coefficient must therefore be formed
before evaluating the force.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from sympy import Matrix, Min, exp, lambdify, sqrt, tanh

from biosym.model.contact import contact_parser
from biosym.model.contact.contact_models.contact_huntcrossley import (
    HuntCrossley as BioSymHuntCrossley,
)


FRICTION_RAMP_A = 0.2
FRICTION_RAMP_K1 = 100.0
FRICTION_RAMP_K2 = 20.0
FRICTION_RAMP_LOCATION = 0.03
FRICTION_RAMP_Q = 50.0
FRICTION_DIRECTION_EPS = 1e-12

# Model B from ``result_HC/p1/friction_ramp_parameter_test.ipynb``.
FRICTION_MODEL_B_A = 0.10
FRICTION_MODEL_B_B = 0.20
FRICTION_MODEL_B_K1 = 200.0
FRICTION_MODEL_B_KM = 15.0
FRICTION_MODEL_B_K2 = 60.0
FRICTION_MODEL_B_LOCATION = 0.6
FRICTION_MODEL_B_ACCELERATION_THRESHOLD = -1.0
FRICTION_MODEL_B_ACCELERATION_TRANSITION = 1.0
FRICTION_MODEL_B_NORMAL_FORCE_THRESHOLD_BW = 0.75
FRICTION_MODEL_B_NORMAL_FORCE_TRANSITION_BW = 0.05
FRICTION_LOADING_RATE_THRESHOLD_BW_PER_S = 5.0
FRICTION_LOADING_RATE_TRANSITION_BW_PER_S = 1.0
HEEL_FORCE_FIT_A = 7.420069391194511
HEEL_FORCE_FIT_V0 = 0.004333660453544265
HEEL_FORCE_FIT_Q = 1.9617596386192722
HEEL_FORCE_AVERAGE_DT = 0.01
# Engage heel friction gradually across early stance.  With this tanh
# parameterization the multiplier is about 0.003 at zero load, 0.5 at
# 35% body weight, and 0.997 at 70% body weight.
HEEL_FRICTION_ACTIVATION_THRESHOLD_BW = 0.35
HEEL_FRICTION_ACTIVATION_TRANSITION_BW = 0.12


def heel_friction_activation(normal_force, body_weight):
    """Smoothly engage heel friction over a small normal-force interval."""
    threshold = HEEL_FRICTION_ACTIVATION_THRESHOLD_BW * body_weight
    transition = HEEL_FRICTION_ACTIVATION_TRANSITION_BW * body_weight
    return 0.5 * (1.0 + jnp.tanh((normal_force - threshold) / transition))


def heel_friction_activation_symbolic(normal_force, body_weight):
    """SymPy form of :func:`heel_friction_activation`."""
    threshold = HEEL_FRICTION_ACTIVATION_THRESHOLD_BW * body_weight
    transition = HEEL_FRICTION_ACTIVATION_TRANSITION_BW * body_weight
    return (1 + tanh((normal_force - threshold) / transition)) / 2


def kinematic_time_derivative(model, expression):
    """Differentiate an expression in BioSym's already-replaced q/qd symbols."""
    expression = Matrix(expression)
    return (
        expression.jacobian(model.coordinates.symbols) * model.speeds.symbols
        + expression.jacobian(model.speeds.symbols) * model.accs.symbols
    )


def friction_model_b_ramp(speed):
    """JAX-compatible three-component monotonic ramp used for negative AP slip."""
    logistic_zero = 1.0 / (
        1.0 + jnp.exp(FRICTION_MODEL_B_K2 * FRICTION_MODEL_B_LOCATION)
    )
    logistic = 1.0 / (
        1.0
        + jnp.exp(
            -FRICTION_MODEL_B_K2 * (speed - FRICTION_MODEL_B_LOCATION)
        )
    )
    sigmoid = (logistic - logistic_zero) / (1.0 - logistic_zero)
    return (
        FRICTION_MODEL_B_A * (1.0 - jnp.exp(-FRICTION_MODEL_B_K1 * speed))
        + FRICTION_MODEL_B_B * (1.0 - jnp.exp(-FRICTION_MODEL_B_KM * speed))
        + (1.0 - FRICTION_MODEL_B_A - FRICTION_MODEL_B_B) * sigmoid
    )


def friction_model_b_ramp_symbolic(speed):
    """SymPy form of :func:`friction_model_b_ramp`."""
    logistic_zero = 1 / (
        1 + exp(FRICTION_MODEL_B_K2 * FRICTION_MODEL_B_LOCATION)
    )
    logistic = 1 / (
        1 + exp(-FRICTION_MODEL_B_K2 * (speed - FRICTION_MODEL_B_LOCATION))
    )
    sigmoid = (logistic - logistic_zero) / (1 - logistic_zero)
    return (
        FRICTION_MODEL_B_A * (1 - exp(-FRICTION_MODEL_B_K1 * speed))
        + FRICTION_MODEL_B_B * (1 - exp(-FRICTION_MODEL_B_KM * speed))
        + (1 - FRICTION_MODEL_B_A - FRICTION_MODEL_B_B) * sigmoid
    )


def directional_friction_ramp(
    speed,
    ap_slip_acceleration,
    transition_velocity,
    normal_force,
    normal_force_threshold,
    normal_force_transition,
    normal_force_rate=None,
    loading_rate_threshold=None,
    loading_rate_transition=None,
    formulation="hybrid",
    heel_normal_force=None,
    body_weight=None,
    is_heel=True,
):
    """Smoothly blend toward model B during rapid deceleration at low load."""
    model_b = friction_model_b_ramp(speed)
    biosym = speed / jnp.sqrt(speed**2 + transition_velocity**2)
    acceleration_weight = 0.5 * (
        1.0
        - jnp.tanh(
            (
                ap_slip_acceleration
                - FRICTION_MODEL_B_ACCELERATION_THRESHOLD
            )
            / FRICTION_MODEL_B_ACCELERATION_TRANSITION
        )
    )
    force_weight = 0.5 * (
        1.0
        - jnp.tanh(
            (normal_force - normal_force_threshold)
            / normal_force_transition
        )
    )
    model_b_weight = acceleration_weight * force_weight
    if formulation in {"option_b_filtered", "heel_force_fit"}:
        loading_weight = 0.5 * (
            1.0
            + jnp.tanh(
                (normal_force_rate - loading_rate_threshold)
                / loading_rate_transition
            )
        )
        model_b_weight = model_b_weight * loading_weight
    if formulation == "heel_force_fit":
        force_smoothing = 1e-6 * body_weight
        positive_heel_force = 0.5 * (
            jnp.sqrt(heel_normal_force**2 + force_smoothing**2)
            + heel_normal_force
        )
        heel_force_bw = positive_heel_force / body_weight
        fitted = (
            HEEL_FORCE_FIT_A * speed
            / (1.0 + speed / HEEL_FORCE_FIT_V0)
            * (heel_force_bw / 0.25) ** HEEL_FORCE_FIT_Q
        )
        selected = jnp.where(is_heel, fitted, biosym)
        return model_b_weight * selected + (1.0 - model_b_weight) * biosym
    return model_b_weight * model_b + (1.0 - model_b_weight) * biosym


def directional_friction_ramp_symbolic(
    speed,
    ap_slip_acceleration,
    transition_velocity,
    normal_force,
    normal_force_threshold,
    normal_force_transition,
    normal_force_rate=None,
    loading_rate_threshold=None,
    loading_rate_transition=None,
    formulation="hybrid",
    heel_normal_force=None,
    body_weight=None,
    is_heel=True,
):
    """SymPy form of :func:`directional_friction_ramp`."""
    model_b = friction_model_b_ramp_symbolic(speed)
    biosym = speed / sqrt(speed**2 + transition_velocity**2)
    acceleration_weight = 0.5 * (
        1
        - tanh(
            (
                ap_slip_acceleration
                - FRICTION_MODEL_B_ACCELERATION_THRESHOLD
            )
            / FRICTION_MODEL_B_ACCELERATION_TRANSITION
        )
    )
    force_weight = 0.5 * (
        1
        - tanh(
            (normal_force - normal_force_threshold)
            / normal_force_transition
        )
    )
    model_b_weight = acceleration_weight * force_weight
    if formulation in {"option_b_filtered", "heel_force_fit"}:
        loading_weight = 0.5 * (
            1
            + tanh(
                (normal_force_rate - loading_rate_threshold)
                / loading_rate_transition
            )
        )
        model_b_weight = model_b_weight * loading_weight
    if formulation == "heel_force_fit":
        force_smoothing = 1e-6 * body_weight
        positive_heel_force = (
            sqrt(heel_normal_force**2 + force_smoothing**2)
            + heel_normal_force
        ) / 2
        heel_force_bw = positive_heel_force / body_weight
        fitted = (
            HEEL_FORCE_FIT_A * speed
            / (1 + speed / HEEL_FORCE_FIT_V0)
            * (heel_force_bw / 0.25) ** HEEL_FORCE_FIT_Q
        )
        selected = fitted if is_heel else biosym
        return model_b_weight * selected + (1 - model_b_weight) * biosym
    return model_b_weight * model_b + (1 - model_b_weight) * biosym


def two_stage_exponential_friction_ramp(speed):
    """JAX-compatible normalized blend of two exponential friction ramps."""
    e1 = FRICTION_RAMP_A * (1.0 - jnp.exp(-FRICTION_RAMP_K1 * speed))
    e1_at_location = FRICTION_RAMP_A * (
        1.0 - jnp.exp(-FRICTION_RAMP_K1 * FRICTION_RAMP_LOCATION)
    )
    e2 = 1.0 - (1.0 - e1_at_location) * jnp.exp(
        -FRICTION_RAMP_K2 * (speed - FRICTION_RAMP_LOCATION)
    )
    switch = 1.0 / (
        1.0 + jnp.exp(-FRICTION_RAMP_Q * (speed - FRICTION_RAMP_LOCATION))
    )
    value = (1.0 - switch) * e1 + switch * e2

    e2_zero = 1.0 - (1.0 - e1_at_location) * jnp.exp(
        FRICTION_RAMP_K2 * FRICTION_RAMP_LOCATION
    )
    switch_zero = 1.0 / (
        1.0 + jnp.exp(FRICTION_RAMP_Q * FRICTION_RAMP_LOCATION)
    )
    value_zero = switch_zero * e2_zero
    return (value - value_zero) / (1.0 - value_zero)


def two_stage_exponential_friction_ramp_symbolic(speed):
    """SymPy form of :func:`two_stage_exponential_friction_ramp`."""
    e1 = FRICTION_RAMP_A * (1 - exp(-FRICTION_RAMP_K1 * speed))
    e1_at_location = FRICTION_RAMP_A * (
        1 - exp(-FRICTION_RAMP_K1 * FRICTION_RAMP_LOCATION)
    )
    e2 = 1 - (1 - e1_at_location) * exp(
        -FRICTION_RAMP_K2 * (speed - FRICTION_RAMP_LOCATION)
    )
    switch = 1 / (1 + exp(-FRICTION_RAMP_Q * (speed - FRICTION_RAMP_LOCATION)))
    value = (1 - switch) * e1 + switch * e2

    e2_zero = 1 - (1 - e1_at_location) * exp(
        FRICTION_RAMP_K2 * FRICTION_RAMP_LOCATION
    )
    switch_zero = 1 / (1 + exp(FRICTION_RAMP_Q * FRICTION_RAMP_LOCATION))
    value_zero = switch_zero * e2_zero
    return (value - value_zero) / (1 - value_zero)


def simbody_composite_modulus(modulus_1: float, modulus_2: float) -> float:
    """Return Simbody's composite plane-strain modulus (both inputs in Pa)."""
    e1_two_thirds = float(modulus_1) ** (2.0 / 3.0)
    e2_two_thirds = float(modulus_2) ** (2.0 / 3.0)
    return (
        e1_two_thirds * e2_two_thirds
        / (e1_two_thirds + e2_two_thirds)
    ) ** 1.5


def effective_radius(feature, other) -> float:
    """Return the Hertz effective radius for a sphere/plane or sphere pair."""
    radius_1 = float(getattr(feature, "radius", 0.0))
    radius_2 = float(getattr(other, "radius", 0.0))
    if radius_1 <= 0.0:
        raise ValueError(
            f"OpenSim Hunt-Crossley contact requires a sphere with positive radius; "
            f"{feature.name!r} has radius {radius_1:g}."
        )
    # A half-space has zero curvature (infinite radius), so R_eff = R_sphere.
    if radius_2 <= 0.0:
        return radius_1
    return radius_1 * radius_2 / (radius_1 + radius_2)


def opensim_hertz_coefficient(
    stiffness: float, feature, other, other_stiffness: float | None = None
) -> float:
    """Convert OpenSim material stiffness (Pa) to Hertz k (N/m**(3/2)).

    OpenSim supplies one stiffness for a contact parameter group and Simbody
    assumes the two contacting materials have that same plane-strain modulus
    unless a second material stiffness is explicitly available.
    """
    modulus_2 = stiffness if other_stiffness is None else other_stiffness
    combined = simbody_composite_modulus(stiffness, modulus_2)
    return (4.0 / 3.0) * np.sqrt(effective_radius(feature, other)) * combined


class HuntCrossley(BioSymHuntCrossley):
    """BioSym contact model with OpenSim/Simbody stiffness semantics.

    ``stiffness`` is a plane-strain modulus in Pa.  ``hertz_coefficients`` is
    populated per contact pair during ``process_eom`` and has units
    N/m**(3/2).
    """

    def __init__(
        self,
        pairs,
        stiffness,
        *args,
        friction_formulation="biosym",
        normal_force_formulation="biosym",
        constant_contact_force=1e-5,
        hertz_smoothing=300.0,
        hunt_crossley_smoothing=50.0,
        loading_rate_threshold_bw_per_s=FRICTION_LOADING_RATE_THRESHOLD_BW_PER_S,
        heel_force_average_dt=HEEL_FORCE_AVERAGE_DT,
        **kwargs,
    ):
        super().__init__(pairs, stiffness, *args, **kwargs)
        if friction_formulation not in {"biosym", "biosym_heel_force_average", "biosym_force_activation", "hybrid", "option_b_filtered", "heel_force_fit"}:
            raise ValueError(
                "friction_formulation must be 'biosym' or 'hybrid', got "
                f"{friction_formulation!r}."
            )
        self.friction_formulation = friction_formulation
        if normal_force_formulation not in {"biosym", "opensim_smooth"}:
            raise ValueError(
                "normal_force_formulation must be 'biosym' or "
                f"'opensim_smooth', got {normal_force_formulation!r}."
            )
        self.normal_force_formulation = normal_force_formulation
        self.constant_contact_force = float(constant_contact_force)
        self.hertz_smoothing = float(hertz_smoothing)
        self.hunt_crossley_smoothing = float(hunt_crossley_smoothing)
        self.loading_rate_threshold_bw_per_s = float(loading_rate_threshold_bw_per_s)
        self.heel_force_average_dt = float(heel_force_average_dt)
        self.material_stiffness = float(stiffness)
        self.combined_modulus = simbody_composite_modulus(
            self.material_stiffness, self.material_stiffness
        )
        self.hertz_coefficients = np.asarray(
            [
                opensim_hertz_coefficient(self.material_stiffness, feature, other)
                for feature, other in self.pairs
            ],
            dtype=float,
        )

    def _pair_normal_force(self, pen, hertz_coefficient=None):
        """Return the scalar Hunt--Crossley normal force for one pair."""
        x = pen["depth"]
        xdot = pen["normal_velocity"]
        coefficient = self.k if hertz_coefficient is None else hertz_coefficient
        if self.normal_force_formulation == "opensim_smooth":
            # Exact SimTK::SmoothSphereHalfSpaceForce normal-force equations.
            # Despite its name, constant_contact_force is added to x**2 in
            # Simbody and is also used in the regularized slip-speed norm.
            hertz_positive = coefficient * sqrt(
                x**2 + self.constant_contact_force
            ) ** 1.5
            hertz = hertz_positive * (
                0.5 + 0.5 * tanh(self.hertz_smoothing * x)
            )
            dissipation = 1 + 1.5 * self.c * xdot
            if self.c == 0.0:
                return hertz
            hunt_crossley_gate = 0.5 + 0.5 * tanh(
                self.hunt_crossley_smoothing
                * (xdot + 2.0 / (3.0 * self.c))
            )
            return hertz * dissipation * hunt_crossley_gate
        x_pos = 0.5 * (sqrt(x**2 + self.eps_depth**2) + x)
        diss = 1 + 1.5 * self.c * xdot
        diss_pos = 0.5 * (sqrt(diss**2 + self.eps_diss**2) + diss)
        return coefficient * x_pos**1.5 * diss_pos + self.grad_bias * x

    def _pair_force(
        self,
        pen,
        hertz_coefficient=None,
        ap_slip_acceleration=0.0,
        foot_normal_force=None,
        foot_normal_force_rate=None,
        heel_normal_force_rate=0.0,
        is_heel=True,
    ):
        """Contact force using a radius-dependent coefficient for this pair."""
        normal = pen["normal"]
        vt = pen["tangential_velocity"]
        f_n = self._pair_normal_force(pen, hertz_coefficient)
        if foot_normal_force is None:
            foot_normal_force = f_n

        speed_sq = vt.dot(vt)
        if self.normal_force_formulation == "opensim_smooth":
            # Exact Simbody smooth-sphere friction magnitude and direction.
            speed = sqrt(speed_sq + self.constant_contact_force)
            speed_ratio = speed / self.v_t
            blend = self.ud + 2 * (self.us - self.ud) / (1 + speed_ratio**2)
            coulomb_magnitude = Min(speed_ratio, 1) * blend
            return (
                f_n * normal
                - f_n * (coulomb_magnitude + self.uv * speed) * vt / speed
            )
        blend = self.ud + 2 * (self.us - self.ud) / (
            1 + speed_sq / self.v_t**2
        )
        speed = sqrt(speed_sq + FRICTION_DIRECTION_EPS**2)
        if self.friction_formulation in {"biosym", "biosym_heel_force_average", "biosym_force_activation"}:
            ramp = speed / sqrt(speed_sq + self.v_t**2)
        else:
            ramp = directional_friction_ramp_symbolic(
                speed,
                ap_slip_acceleration,
                self.v_t,
                foot_normal_force,
                self.model_b_normal_force_threshold,
                self.model_b_normal_force_transition,
                foot_normal_force_rate,
                self.loading_rate_threshold,
                self.loading_rate_transition,
                self.friction_formulation,
                f_n,
                self.body_weight,
                is_heel,
            )
        coulomb = blend * ramp * vt / speed
        friction_normal_force = f_n
        if self.friction_formulation == "biosym_heel_force_average" and is_heel:
            estimated_average = f_n - 0.5 * self.heel_force_average_dt * heel_normal_force_rate
            smoothing = 1e-6 * self.body_weight
            friction_normal_force = (
                sqrt(estimated_average**2 + smoothing**2) + estimated_average
            ) / 2
        elif self.friction_formulation == "biosym_force_activation" and is_heel:
            friction_normal_force = f_n * heel_friction_activation_symbolic(
                foot_normal_force, self.body_weight
            )
        return f_n * normal - friction_normal_force * (coulomb + self.uv * vt)

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
        force_exprs, pos_exprs = [], []
        pair_force_exprs, pair_normal_exprs, pair_slip_exprs = [], [], []
        contrib_body_slot, contrib_fk_idx = [], []

        def add_contribution(force_expr, geometry):
            force_exprs.append(force_expr)
            pos_exprs.append(geometry.pos_expr)
            contrib_body_slot.append(self._body_index(geometry.parent_body))
            contrib_fk_idx.append(body_keys.index(geometry.parent_body))

        penetrations = [feature.penetration(other) for feature, other in self.pairs]
        pair_normal_forces = [
            self._pair_normal_force(penetration, self.hertz_coefficients[index])
            for index, penetration in enumerate(penetrations)
        ]
        pair_normal_force_rates = [
            kinematic_time_derivative(model, Matrix([force]))[0]
            for force in pair_normal_forces
        ]
        foot_normal_forces = {}
        for normal_force, (feature, _) in zip(pair_normal_forces, self.pairs):
            foot_normal_forces[feature.parent_body] = (
                foot_normal_forces.get(feature.parent_body, 0) + normal_force
            )
        foot_normal_force_rates = {
            body: kinematic_time_derivative(model, Matrix([force]))[0]
            for body, force in foot_normal_forces.items()
        }

        for pair_index, ((feature, other), penetration) in enumerate(
            zip(self.pairs, penetrations)
        ):
            ap_slip_acceleration = kinematic_time_derivative(
                model, Matrix([penetration["tangential_velocity"][0]])
            )[0]
            force_on_feature = self._pair_force(
                penetration,
                self.hertz_coefficients[pair_index],
                ap_slip_acceleration,
                foot_normal_forces[feature.parent_body],
                foot_normal_force_rates[feature.parent_body],
                pair_normal_force_rates[pair_index],
                "heel" in feature.name.lower(),
            )
            pair_force_exprs.append(force_on_feature.T)
            pair_normal_exprs.append(penetration["normal"].T)
            pair_slip_exprs.append(penetration["tangential_velocity"].T)
            add_contribution(force_on_feature, feature)
            if other.parent_body != "ground_frame":
                add_contribution(-force_on_feature, other)

        force_matrix = Matrix.vstack(*[force.T for force in force_exprs])
        pos_matrix = Matrix.vstack(*[position.T for position in pos_exprs])
        self.force_fn = lambdify(
            model._symbols, force_matrix, modules="jax", cse=True, docstring_limit=2
        )
        self.pos_fn = lambdify(
            model._symbols, pos_matrix, modules="jax", cse=True, docstring_limit=2
        )
        self.pair_force_fn = lambdify(
            model._symbols,
            Matrix.vstack(*pair_force_exprs),
            modules="jax",
            cse=True,
            docstring_limit=2,
        )
        self.pair_normal_fn = lambdify(
            model._symbols,
            Matrix.vstack(*pair_normal_exprs),
            modules="jax",
            cse=True,
            docstring_limit=2,
        )
        self.pair_slip_fn = lambdify(
            model._symbols,
            Matrix.vstack(*pair_slip_exprs),
            modules="jax",
            cse=True,
            docstring_limit=2,
        )
        self._contrib_body_slot = np.asarray(contrib_body_slot)
        self._contrib_fk_idx = np.asarray(contrib_fk_idx)

    def contact_slip_and_friction(self, states, constants):
        """Return per-pair tangential velocity, friction, and normal force."""
        s_flat = states.filter("model").flatten()
        c_flat = constants.filter("model").flatten()
        slip = jnp.asarray(self.pair_slip_fn(*s_flat, *c_flat)).reshape(-1, 3)
        normal = jnp.asarray(self.pair_normal_fn(*s_flat, *c_flat)).reshape(-1, 3)
        force = jnp.asarray(self.pair_force_fn(*s_flat, *c_flat)).reshape(-1, 3)
        normal_force = jnp.sum(force * normal, axis=1)
        friction = force - normal_force[:, None] * normal
        return slip, friction, normal_force


_ORIGINAL_XML_FORCE_PARAM_READER = contact_parser._read_xml_force_params


def _read_xml_force_params(force_law, attr):
    """Extend BioSym's Hunt-Crossley XML reader with OpenSim smoothing fields."""
    params = _ORIGINAL_XML_FORCE_PARAM_READER(force_law, attr)
    if force_law != "huntcrossley":
        return params
    return params + (
        ("normal_force_formulation", attr("normal_force_formulation", "biosym")),
        ("constant_contact_force", float(attr("constant_contact_force", 1e-5))),
        ("hertz_smoothing", float(attr("hertz_smoothing", 300.0))),
        ("hunt_crossley_smoothing", float(attr("hunt_crossley_smoothing", 50.0))),
    )


def install(friction_formulation="biosym", loading_rate_threshold_bw_per_s=FRICTION_LOADING_RATE_THRESHOLD_BW_PER_S, heel_force_average_dt=HEEL_FORCE_AVERAGE_DT) -> None:
    """Make BioSym's standard XML and OSIM parser construct this local class."""
    contact_parser._FORCE_LAWS["huntcrossley"] = (
        lambda pairs, parameters: HuntCrossley(
            pairs, friction_formulation=friction_formulation,
            loading_rate_threshold_bw_per_s=loading_rate_threshold_bw_per_s,
            heel_force_average_dt=heel_force_average_dt,
            **parameters
        )
    )
    contact_parser._read_xml_force_params = _read_xml_force_params
