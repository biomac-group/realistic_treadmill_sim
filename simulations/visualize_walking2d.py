import argparse
import csv
import os
import re
from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from biosym.model import model
from biosym.utils.states import States
from biosym.visualization import stickfigure

from src.gc_model_huntcrossley_treadmill import (
    HuntCrossleyTreadmill,
    MultiHuntCrossleyTreadmill,
)
from src.contact_huntcrossley import install as install_huntcrossley_contact
from src.gait_data import DEFAULT_GAIT_DATA_FILE, get_belt_speeds, get_duration
from src.treadmill_controller import (
    AP_GRF_FILTER_CUTOFF_HZ,
    cubic_bridge_post_hs,
    periodic_butterworth_filtfilt,
)


DESIRED_GRF_COLUMNS = {
    "grf_x_r": "grf_x_r_mean",
    "grf_y_r": "grf_y_r_mean",
    "grf_x_l": "grf_x_l_mean",
    "grf_y_l": "grf_y_l_mean",
}

def load_model_with_treadmill_contact(
    force_rebuild=True,
    model_file=None,
):
    install_huntcrossley_contact()
    if model_file is None:
        raise ValueError("A Hunt-Crossley model_file is required.")
    biosym_model = model.load_model(str(model_file), force_rebuild=force_rebuild)
    if biosym_model.gc_model.__class__.__name__ == "HuntCrossley":
        biosym_model.gc_model = HuntCrossleyTreadmill(biosym_model.gc_model)
    elif biosym_model.gc_model.__class__.__name__ == "MultiContact":
        biosym_model.gc_model = MultiHuntCrossleyTreadmill(biosym_model.gc_model)
    else:
        raise ValueError(
            "Saved treadmill result uses unsupported contact model "
            f"{type(biosym_model.gc_model).__name__}."
        )
    biosym_model.gc_model.process_eom(biosym_model)
    biosym_model._register_contact_model(biosym_model.gc_model)
    return biosym_model


def load_solution(result_file):
    with open(result_file, "rb") as handle:
        payload = cloudpickle.load(handle)
    return payload


def extract_solution(payload):
    (states_dict, globals_dict), info, settings = payload
    return states_dict, globals_dict, info, settings


def load_mot_table(file_path):
    with open(file_path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]

    endheader_index = lines.index("endheader")
    columns = lines[endheader_index + 1].split()
    data = np.vstack([np.fromstring(line, sep="\t") for line in lines[endheader_index + 2 :]])
    if data.shape[1] != len(columns):
        data = np.vstack([np.fromstring(line, sep=" ") for line in lines[endheader_index + 2 :]])
    return columns, data


def resample_series(values, target_count):
    source = np.asarray(values, dtype=float).reshape(-1)
    if source.size == target_count:
        return source.copy()

    source_phase = np.arange(source.size, dtype=float) / max(source.size - 1, 1)
    target_phase = np.arange(target_count, dtype=float) / max(target_count - 1, 1)
    return np.interp(target_phase, source_phase, source)


def find_track_grf_file_in_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    collocation = payload.get("collocation", payload)
    for objective in collocation.get("objectives", []):
        if objective.get("name") != "track_grf":
            continue
        args = objective.get("args", {})
        candidate = args.get("presegmented_file")
        if candidate:
            return candidate
    return None


def infer_participant_from_tracking_config(config_path):
    if config_path is None:
        return None
    match = re.search(r"p(\d+)", os.path.basename(config_path))
    if match is None:
        return None
    return int(match.group(1))


def infer_participant_from_path(path):
    if path is None:
        return None
    match = re.search(r"(?:^|[/\\])p(\d+)(?:[/\\]|$)", str(path))
    if match is None:
        return None
    return int(match.group(1))


def infer_speed_from_tracking_config(config_path):
    if config_path is None:
        return None
    match = re.search(r"p\d+_(\d+)_(\d+)\.ya?ml$", os.path.basename(config_path))
    if match is None:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def infer_speed_from_result_path(path):
    if path is None:
        return None
    match = re.search(r"_(\d+)_(\d+)(?:[^0-9].*)?\.pkl$", os.path.basename(str(path)))
    if match is None:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def infer_speed_from_settings(settings):
    metadata = get_treadmill_controller_metadata(settings)
    target_speed = metadata.get("target_speed")
    if target_speed is None:
        return None
    return abs(float(target_speed))


def infer_desired_grf_path(speed, participant=None, output_dir="data/opensim_exports"):
    if participant is not None:
        token = f"{abs(float(speed)):.1f}".replace(".", "p")
        participant_path = os.path.join(output_dir, f"p{int(participant)}_grf_mean_speed_{token}.mot")
        if os.path.exists(participant_path):
            return participant_path

    candidates = []
    for entry in os.listdir(output_dir):
        match = re.fullmatch(r"(?:p\d+_)?grf_mean_speed_(\d+)p(\d+)\.mot", entry)
        if not match:
            continue
        candidate_speed = float(f"{match.group(1)}.{match.group(2)}")
        candidates.append((abs(candidate_speed - speed), candidate_speed, os.path.join(output_dir, entry)))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def resolve_desired_grf_path(args, tracking_speed, participant):
    if args.desired_grf is not None:
        return args.desired_grf
    if args.tracking_config is not None:
        config_path = find_track_grf_file_in_config(args.tracking_config)
        if config_path is not None:
            return config_path
    return infer_desired_grf_path(tracking_speed, participant=participant)


def load_desired_grf(desired_grf_path, target_count):
    if desired_grf_path is None or not os.path.exists(desired_grf_path):
        return None

    columns, data = load_mot_table(desired_grf_path)
    column_lookup = {name: index for index, name in enumerate(columns)}
    desired = {}
    for key, column_name in DESIRED_GRF_COLUMNS.items():
        if column_name not in column_lookup:
            return None
        desired[key] = resample_series(data[:, column_lookup[column_name]], target_count)
    return desired


JOINT_REFERENCE_COLUMNS = {
    "q_hip_r": "hip_flexion_r",
    "q_knee_r": "knee_angle_r",
    "q_ankle_r": "ankle_angle_r",
    "q_hip_l": "hip_flexion_l",
    "q_knee_l": "knee_angle_l",
    "q_ankle_l": "ankle_angle_l",
}


def resolve_joint_angle_reference_path(settings, tracking_speed):
    if isinstance(settings, dict):
        for objective in settings.get("objectives", []):
            if objective.get("name") != "track_angles":
                continue
            candidate = objective.get("args", {}).get("presegmented_file")
            if candidate and os.path.exists(candidate):
                return candidate

    cadence = "natural" if abs(float(tracking_speed)) <= 1.4 else "fast"
    candidate = os.path.join(
        "data",
        "opensim_exports",
        "winter_1987",
        f"winter_1987_{cadence}_cadence_kinematics_mean_var_rad.mot",
    )
    return candidate if os.path.exists(candidate) else None


def load_desired_joint_angles(reference_path, target_count):
    if reference_path is None or not os.path.exists(reference_path):
        return None

    columns, data = load_mot_table(reference_path)
    lookup = {name: index for index, name in enumerate(columns)}
    desired = {}
    for state_name, column_base in JOINT_REFERENCE_COLUMNS.items():
        mean_column = f"{column_base}_mean"
        variance_column = f"{column_base}_var"
        if mean_column not in lookup or variance_column not in lookup:
            return None
        desired[state_name] = {
            "mean": resample_series(data[:, lookup[mean_column]], target_count),
            "std": np.sqrt(
                np.maximum(
                    resample_series(data[:, lookup[variance_column]], target_count),
                    0.0,
                )
            ),
        }
    return desired


def resolve_participant(args):
    if args.participant is not None:
        return args.participant
    participant = infer_participant_from_tracking_config(args.tracking_config)
    if participant is not None:
        return participant
    return infer_participant_from_path(args.result)


def resolve_tracking_speed(args, globals_dict, settings):
    if args.speed is not None:
        return float(args.speed)
    tracking_speed = infer_speed_from_tracking_config(args.tracking_config)
    if tracking_speed is not None:
        return tracking_speed
    tracking_speed = infer_speed_from_result_path(args.result)
    if tracking_speed is not None:
        return tracking_speed
    tracking_speed = infer_speed_from_settings(settings)
    if tracking_speed is not None:
        return tracking_speed
    return abs(float(globals_dict.speed))


def load_desired_belt_speed(gait_data_path, participant, target_speed, target_count):
    if gait_data_path is None or not os.path.exists(gait_data_path) or participant is None:
        return None

    return get_belt_speeds(gait_data_path, participant, target_speed, target_count)


def get_treadmill_controller_metadata(settings):
    if not isinstance(settings, dict):
        return {}
    metadata = settings.get("treadmill_controller", {})
    if metadata:
        return metadata

    for constraint in settings.get("constraints", []):
        if "treadmill" not in str(constraint.get("name", "")):
            continue
        args = constraint.get("args", {})
        if args:
            return args
    return {}


def diagnose_foot_strike(
    biosym_model,
    states_dict,
    duration=None,
    toe_fraction_threshold=0.80,
    contact_threshold_bw=0.10,
):
    """Classify right and left touchdown from heel/toe vertical contact forces."""
    if isinstance(states_dict, States) and hasattr(biosym_model.gc_model, "pairs"):
        cp_names = [feature.name for feature, _ in biosym_model.gc_model.pairs]
    elif isinstance(states_dict, States) and hasattr(biosym_model.gc_model, "models"):
        cp_names = [
            feature.name
            for contact in biosym_model.gc_model.models
            for feature, _ in contact.pairs
        ]
    else:
        cp_names = list(biosym_model.gc_model.cps)
    required = {f"{point}_{side}" for point in ("heel", "toe") for side in ("r", "l")}
    if not required.issubset(cp_names):
        return None

    if isinstance(states_dict, States):
        mass = float(np.asarray(biosym_model.default_constants.mass).sum())
    else:
        mass = float(sum(
            states_dict.constants.model[index]
            for index, name in enumerate(biosym_model.constants)
            if name.startswith("m_")
        ))
    force_threshold = contact_threshold_bw * mass * 9.81
    if isinstance(states_dict, States):
        states_dict = reconstruct_accelerations(states_dict, duration)

        def sphere_forces(state):
            full_state = state.replace(
                tau=jnp.zeros(biosym_model.tau.n),
                ext_forces=jnp.zeros(biosym_model.ext_forces.n),
                ext_torques=jnp.zeros(biosym_model.ext_torques.n),
            )
            return biosym_model.gc_model._contact_positions_and_forces(
                full_state, biosym_model.default_constants
            )[1]

        vertical_forces = np.asarray(jax.vmap(sphere_forces)(states_dict))[:, :, 1]
    else:
        vertical_forces = []
        for node in range(len(states_dict.states.model)):
            h = None if states_dict.states.h is None else states_dict.states.h[node]
            node_states = States(
                model=states_dict.states.model[node],
                gc_model=states_dict.states.gc_model[node],
                actuator_model=states_dict.states.actuator_model[node],
                h=h,
            )
            vertical_forces.append(np.asarray(
                biosym_model.gc_model.get_cp_forces(
                    node_states, states_dict.constants, biosym_model
                )
            )[:, 1])
        vertical_forces = np.asarray(vertical_forces)

    diagnosis = {}
    for side, side_name in (("r", "Right"), ("l", "Left")):
        heel_force = vertical_forces[:, cp_names.index(f"heel_{side}")]
        toe_force = vertical_forces[:, cp_names.index(f"toe_{side}")]
        total_force = heel_force + toe_force
        in_contact = total_force >= force_threshold
        previous_contact = np.roll(in_contact, 1)
        touchdown_candidates = np.flatnonzero(in_contact & ~previous_contact)
        touchdown_node = (
            int(touchdown_candidates[0])
            if touchdown_candidates.size
            else int(np.argmax(total_force))
        )
        denominator = total_force[touchdown_node]
        toe_fraction = float(toe_force[touchdown_node] / denominator) if denominator > 0 else np.nan
        landing = "toe" if toe_fraction >= toe_fraction_threshold else "heel/midfoot"
        diagnosis[side] = {
            "side": side_name,
            "landing": landing,
            "toe_fraction": toe_fraction,
            "touchdown_node": touchdown_node,
        }
    return diagnosis


def foot_strike_label(diagnosis):
    if not diagnosis:
        return "Foot strike: unavailable"
    labels = []
    for side in ("r", "l"):
        result = diagnosis[side]
        labels.append(
            f"{result['side']}: {result['landing']} "
            f"({100 * result['toe_fraction']:.0f}% toe at touchdown)"
        )
    return "Foot strike | " + " | ".join(labels)


def create_summary_figure(
    biosym_model,
    states_dict,
    globals_dict,
    desired_grf=None,
    desired_belt_speed=None,
    experimental_duration=None,
    treadmill_controller_metadata=None,
    controller_ap_filter_cutoff_hz=AP_GRF_FILTER_CUTOFF_HZ,
    foot_strike_diagnosis=None,
):
    if isinstance(states_dict, States):
        states_dict = reconstruct_accelerations(states_dict, float(globals_dict.dur))
        treadmill_states = np.asarray(states_dict.gc_model, dtype=np.float64)

        def contact_force(state):
            full_state = state.replace(
                tau=jnp.zeros(biosym_model.tau.n),
                ext_forces=jnp.zeros(biosym_model.ext_forces.n),
                ext_torques=jnp.zeros(biosym_model.ext_torques.n),
            )
            return biosym_model.run["gc_model"](
                full_state, biosym_model.default_constants
            )[0]

        contact_forces = np.asarray(jax.vmap(contact_force)(states_dict))
        grf_x_r, grf_y_r = contact_forces[:, 0, 0], contact_forces[:, 0, 1]
        grf_x_l, grf_y_l = contact_forces[:, 1, 0], contact_forces[:, 1, 1]
        n_nodes = len(states_dict)
    else:
        model_states = np.asarray(states_dict.states.model, dtype=np.float64)
        treadmill_states = np.asarray(states_dict.states.gc_model, dtype=np.float64)
        grf_x_r = model_states[:, biosym_model.state_vector.index("f_foot_r_x")]
        grf_y_r = model_states[:, biosym_model.state_vector.index("f_foot_r_y")]
        grf_x_l = model_states[:, biosym_model.state_vector.index("f_foot_l_x")]
        grf_y_l = model_states[:, biosym_model.state_vector.index("f_foot_l_y")]
        n_nodes = model_states.shape[0]
    node_index = np.arange(n_nodes)
    average_treadmill_speed = float(np.mean(treadmill_states))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    line_belt_r = axes[0].plot(node_index, -treadmill_states[:, 0], label="Right belt speed")[0]
    line_belt_l = axes[0].plot(node_index, -treadmill_states[:, 1], label="Left belt speed")[0]
    if desired_belt_speed is not None:
        axes[0].plot(
            node_index,
            desired_belt_speed["right"],
            linestyle=":",
            linewidth=2,
            color=line_belt_r.get_color(),
            label="Desired right belt speed",
        )
        axes[0].plot(
            node_index,
            desired_belt_speed["left"],
            linestyle=":",
            linewidth=2,
            color=line_belt_l.get_color(),
            label="Desired left belt speed",
        )
    axes[0].set_ylabel("Belt speed (m/s)")
    axes[0].set_title("Treadmill States")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    line_grf_x_r = axes[1].plot(node_index, grf_x_r, label="GRF right x")[0]
    line_grf_y_r = axes[1].plot(node_index, grf_y_r, label="GRF right y")[0]
    line_grf_x_l = axes[1].plot(node_index, grf_x_l, label="GRF left x")[0]
    line_grf_y_l = axes[1].plot(node_index, grf_y_l, label="GRF left y")[0]
    treadmill_controller_metadata = treadmill_controller_metadata or {}
    if abs(float(treadmill_controller_metadata.get("k_fx", 0.0))) > 0.0:
        dt = float(globals_dict.dur) / (n_nodes - 1)

        default_interpolation_samples = int(
            treadmill_controller_metadata.get("post_hs_interpolation_samples", 0)
        )
        interpolation_samples_r = int(
            treadmill_controller_metadata.get(
                "post_hs_interpolation_samples_r", default_interpolation_samples
            )
        )
        interpolation_samples_l = int(
            treadmill_controller_metadata.get(
                "post_hs_interpolation_samples_l", default_interpolation_samples
            )
        )
        heel_strike_index_r = int(
            treadmill_controller_metadata.get("heel_strike_index_r", 0)
        )
        heel_strike_index_l = int(
            treadmill_controller_metadata.get(
                "heel_strike_index_l", (n_nodes - 1) // 2
            )
        )

        def controller_input(values, heel_strike_index, interpolation_samples):
            periodic = np.asarray(values, dtype=float)[:-1]
            interpolated = np.asarray(
                cubic_bridge_post_hs(
                    periodic, heel_strike_index, interpolation_samples
                )
            )
            return np.r_[interpolated, interpolated[0]]

        if interpolation_samples_r > 0:
            axes[1].plot(
                node_index,
                controller_input(
                    grf_x_r, heel_strike_index_r, interpolation_samples_r
                ),
                linestyle="--",
                linewidth=2,
                color=line_grf_x_r.get_color(),
                label="Controller input right x (interpolation only)",
            )
        if interpolation_samples_l > 0:
            axes[1].plot(
                node_index,
                controller_input(
                    grf_x_l, heel_strike_index_l, interpolation_samples_l
                ),
                linestyle="--",
                linewidth=2,
                color=line_grf_x_l.get_color(),
                label="Controller input left x (interpolation only)",
            )
    if desired_grf is not None:
        axes[1].plot(
            node_index,
            desired_grf["grf_x_r"],
            linestyle=":",
            linewidth=2,
            color=line_grf_x_r.get_color(),
            label="Experimental GRF right x",
        )
        axes[1].plot(
            node_index,
            desired_grf["grf_y_r"],
            linestyle=":",
            linewidth=2,
            color=line_grf_y_r.get_color(),
            label="Experimental GRF right y",
        )
        axes[1].plot(
            node_index,
            desired_grf["grf_x_l"],
            linestyle=":",
            linewidth=2,
            color=line_grf_x_l.get_color(),
            label="Experimental GRF left x",
        )
        axes[1].plot(
            node_index,
            desired_grf["grf_y_l"],
            linestyle=":",
            linewidth=2,
            color=line_grf_y_l.get_color(),
            label="Experimental GRF left y",
        )
    axes[1].set_xlabel("Node")
    axes[1].set_ylabel("Force (N)")
    axes[1].set_title(f"Ground Reaction Forces\n{foot_strike_label(foot_strike_diagnosis)}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    simulated_duration = float(globals_dict.dur)
    title_parts = [
        "walking2d summary",
        f"sim dur={simulated_duration:.3f} s",
        f"avg treadmill speed={average_treadmill_speed:.3f} m/s",
    ]
    if experimental_duration is not None:
        duration_delta = simulated_duration - float(experimental_duration)
        title_parts.insert(2, f"exp dur={float(experimental_duration):.3f} s")
        title_parts.insert(3, f"delta dur={duration_delta:+.3f} s")
    if "delay" in treadmill_controller_metadata:
        title_parts.append(f"GRF delay={int(treadmill_controller_metadata['delay'])} samples")
    if "error_delay_samples" in treadmill_controller_metadata:
        title_parts.append(
            f"error delay={int(treadmill_controller_metadata['error_delay_samples'])} samples"
        )
    fig.suptitle(" | ".join(title_parts))
    fig.tight_layout()
    return fig


def joint_angle_output_path(summary_path):
    root, _ = os.path.splitext(str(summary_path))
    return f"{root}_joint_angles.png"


def slip_friction_output_path(summary_path):
    root, _ = os.path.splitext(str(summary_path))
    return f"{root}_slip_friction.png"


def contact_diagnostics_output_path(result_path):
    root, _ = os.path.splitext(str(result_path))
    return f"{root}_contact_diagnostics.csv"


def reconstruct_accelerations(states_dict, duration):
    """Restore the backward-Euler qdd used by the OCP from a saved qd trajectory."""
    if states_dict.qdd is not None:
        return states_dict
    if duration is None:
        # The friction selector uses only the sign of acceleration, which is
        # invariant to multiplication by the positive gait-cycle duration.
        duration = 1.0
    qd = np.asarray(states_dict.qd, dtype=float)
    if len(qd) < 2:
        qdd = np.zeros_like(qd)
    else:
        dt = float(duration) / (len(qd) - 1)
        qdd = np.empty_like(qd)
        qdd[1:] = np.diff(qd, axis=0) / dt
        # The last node duplicates the start of the periodic gait cycle.
        previous = qd[-2] if len(qd) > 2 else qd[-1]
        qdd[0] = (qd[0] - previous) / dt
    return states_dict.replace(qdd=jnp.asarray(qdd))


def evaluate_contact_diagnostics(biosym_model, states_dict, duration=None):
    """Evaluate per-contact slip, friction, and normal force at every node."""
    if not isinstance(states_dict, States):
        raise TypeError("Slip/friction plotting requires BioSym States output.")
    states_dict = reconstruct_accelerations(states_dict, duration)

    contact = biosym_model.gc_model
    if hasattr(contact, "pairs"):
        pairs = contact.pairs
    elif hasattr(contact, "models"):
        pairs = [pair for group in contact.models for pair in group.pairs]
    else:
        raise TypeError(
            f"Contact model {type(contact).__name__} does not expose contact pairs."
        )
    side_slots = np.asarray(
        [0 if feature.parent_body.endswith("_r") else 1 for feature, _ in pairs]
    )
    names = [feature.name for feature, _ in pairs]

    def evaluate(state):
        full_state = state.replace(
            tau=jnp.zeros(biosym_model.tau.n),
            ext_forces=jnp.zeros(biosym_model.ext_forces.n),
            ext_torques=jnp.zeros(biosym_model.ext_torques.n),
        )
        if hasattr(contact, "contact_slip_and_friction"):
            return contact.contact_slip_and_friction(
                full_state, biosym_model.default_constants
            )
        diagnostics = [
            group.contact_slip_and_friction(
                full_state, biosym_model.default_constants
            )
            for group in contact.models
        ]
        return tuple(jnp.concatenate(values, axis=0) for values in zip(*diagnostics))

    slip, friction, normal_force = jax.vmap(evaluate)(states_dict)
    return (
        names,
        side_slots,
        np.asarray(slip),
        np.asarray(friction),
        np.asarray(normal_force),
    )


def save_contact_diagnostics_csv(
    biosym_model, states_dict, output_path, duration=None
):
    """Write the values entering the friction law for every contact and node."""
    names, side_slots, slip, friction, normal_force = evaluate_contact_diagnostics(
        biosym_model, states_dict, duration
    )
    contact = biosym_model.gc_model
    if hasattr(contact, "_vt_by_pair"):
        transition_velocity = np.asarray(contact._vt_by_pair, dtype=float)
    else:
        transition_velocity = np.full(len(names), float(contact.v_t))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "node",
                "gait_cycle_percent",
                "side",
                "contact",
                "normal_force_N",
                "vs_m_per_s",
                "vs_x_m_per_s",
                "vt_m_per_s",
                "friction_force_x_N",
            ]
        )
        denominator = max(len(states_dict) - 1, 1)
        for node in range(len(states_dict)):
            for index, name in enumerate(names):
                writer.writerow(
                    [
                        node,
                        100.0 * node / denominator,
                        "right" if side_slots[index] == 0 else "left",
                        name,
                        normal_force[node, index],
                        np.linalg.norm(slip[node, index]),
                        slip[node, index, 0],
                        transition_velocity[index],
                        friction[node, index, 0],
                    ]
                )
    return output_path


def create_slip_friction_figure(biosym_model, states_dict, duration=None):
    """Plot force-weighted fore-aft contact slip and total foot friction."""
    names, side_slots, slip, friction, normal_force = evaluate_contact_diagnostics(
        biosym_model, states_dict, duration
    )
    normal_force = np.maximum(normal_force, 0.0)
    gait_cycle = np.linspace(0.0, 100.0, len(states_dict))

    weighted_slip = np.zeros((len(states_dict), 2))
    total_friction = np.zeros((len(states_dict), 2))
    for side in (0, 1):
        mask = side_slots == side
        weights = normal_force[:, mask]
        denominator = weights.sum(axis=1)
        numerator = (slip[:, mask, 0] * weights).sum(axis=1)
        weighted_slip[:, side] = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-8,
        )
        total_friction[:, side] = friction[:, mask, 0].sum(axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    for side, label, color in ((0, "Right", "tab:blue"), (1, "Left", "tab:orange")):
        axes[0].plot(
            gait_cycle, weighted_slip[:, side], label=label, color=color, linewidth=2
        )
        axes[1].plot(
            gait_cycle, total_friction[:, side], label=label, color=color, linewidth=2
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Slip velocity (m/s)")
    axes[0].set_title("Normal-force-weighted fore-aft contact slip")
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel("Friction force (N)")
    axes[1].set_title("Total fore-aft foot friction")

    for index, name in enumerate(names):
        side = "Right" if side_slots[index] == 0 else "Left"
        label = f"{side} {name}"
        loaded_slip = np.where(
            normal_force[:, index] > 1.0,
            slip[:, index, 0],
            np.nan,
        )
        axes[2].plot(gait_cycle, loaded_slip, label=label, linewidth=1.6)
        axes[3].plot(
            gait_cycle, friction[:, index, 0], label=label, linewidth=1.6
        )

    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("Slip velocity (m/s)")
    axes[2].set_title("Per-contact fore-aft slip while loaded")
    axes[3].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[3].set_ylabel("Friction force (N)")
    axes[3].set_xlabel("Gait cycle (%)")
    axes[3].set_title("Per-contact fore-aft friction")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def create_joint_angle_figure(
    biosym_model,
    states_dict,
    desired_joint_angles=None,
    foot_strike_diagnosis=None,
):
    """Plot optimized lower-limb angles and net joint moments."""
    if isinstance(states_dict, States):
        model_states = np.asarray(states_dict.q, dtype=np.float64)

        def actuator_force(state):
            full_state = state.replace(
                qdd=jnp.zeros(biosym_model.accs.n),
                tau=jnp.zeros(biosym_model.tau.n),
                ext_forces=jnp.zeros(biosym_model.ext_forces.n),
                ext_torques=jnp.zeros(biosym_model.ext_torques.n),
            )
            return biosym_model.run["actuator_model"](
                full_state, biosym_model.default_constants
            )

        joint_moments = np.asarray(jax.vmap(actuator_force)(states_dict)).reshape(
            len(states_dict), -1
        )
        coordinate_names = list(biosym_model.coordinates.names)
        moment_names = list(biosym_model.tau.names)
    else:
        model_states = np.asarray(states_dict.states.model, dtype=np.float64)
        joint_moments = np.asarray(
            biosym_model.run["actuator_model"](
                states_dict.states,
                states_dict.constants,
            ),
            dtype=np.float64,
        )
        coordinate_names = list(biosym_model.state_vector)
        moment_names = list(biosym_model.forces["names"])
    gait_cycle = np.linspace(0.0, 100.0, model_states.shape[0])
    joint_rows = (
        (("q_hip_r", "t_hip_r", "Right hip"), ("q_hip_l", "t_hip_l", "Left hip")),
        (("q_knee_r", "t_knee_r", "Right knee"), ("q_knee_l", "t_knee_l", "Left knee")),
        (("q_ankle_r", "t_ankle_r", "Right ankle"), ("q_ankle_l", "t_ankle_l", "Left ankle")),
    )

    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True)
    for row_index, sides in enumerate(joint_rows):
        for side_index, (state_name, moment_name, label) in enumerate(sides):
            angle_axis = axes[row_index, side_index * 2]
            moment_axis = axes[row_index, side_index * 2 + 1]

            if state_name not in coordinate_names:
                angle_axis.set_visible(False)
            else:
                state_index = coordinate_names.index(state_name)
                angle_axis.plot(
                    gait_cycle,
                    np.rad2deg(model_states[:, state_index]),
                    linewidth=2,
                    label="Simulated",
                )
            if desired_joint_angles is not None and state_name in desired_joint_angles:
                reference = desired_joint_angles[state_name]
                reference_mean = np.rad2deg(reference["mean"])
                reference_std = np.rad2deg(reference["std"])
                angle_axis.plot(
                    gait_cycle,
                    reference_mean,
                    color="black",
                    linestyle="--",
                    linewidth=1.8,
                    label="Experimental mean",
                )
                angle_axis.fill_between(
                    gait_cycle,
                    reference_mean - reference_std,
                    reference_mean + reference_std,
                    color="black",
                    alpha=0.12,
                    linewidth=0,
                    label="Experimental ±1 SD",
                )
            angle_axis.set_title(f"{label} angle")
            angle_axis.set_ylabel("Angle (deg)")
            angle_axis.grid(True, alpha=0.3)
            angle_axis.legend(fontsize="small")

            resolved_moment_name = moment_name
            if resolved_moment_name not in moment_names:
                resolved_moment_name = moment_name.replace("t_", "M_", 1)
            if resolved_moment_name not in moment_names:
                moment_axis.set_visible(False)
                continue
            moment_index = moment_names.index(resolved_moment_name)
            moment_axis.plot(
                gait_cycle,
                joint_moments[:, moment_index],
                linewidth=2,
                color="tab:red",
                label="Net moment",
            )
            moment_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            moment_axis.set_title(f"{label} moment")
            moment_axis.set_ylabel("Moment (N·m)")
            moment_axis.grid(True, alpha=0.3)
            moment_axis.legend(fontsize="small")

    for axis in axes[-1, :]:
        axis.set_xlabel("Gait cycle (%)")
    fig.suptitle(
        "Optimized joint angles and net moments (gait2d sign convention)\n"
        + foot_strike_label(foot_strike_diagnosis)
    )
    fig.tight_layout()
    return fig


def save_figure(fig, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a saved script2d walking result with summary plots and a biosym stickfigure."
    )
    parser.add_argument("--result", default="result/walking2d.pkl", help="Path to the saved result pickle.")
    parser.add_argument(
        "--desired-grf",
        default=None,
        help="Optional path to a desired GRF MOT file to overlay as dotted lines.",
    )
    parser.add_argument(
        "--tracking-config",
        default=None,
        help="Optional tracking YAML file from which to read the track_grf presegmented_file.",
    )
    parser.add_argument(
        "--belt-speeds",
        default=str(DEFAULT_GAIT_DATA_FILE),
        help="Optional gait_data.csv file with desired belt speeds.",
    )
    parser.add_argument(
        "--participant",
        type=int,
        default=None,
        help="Participant index for desired belt speeds. If omitted, infer from tracking config filename like walking_tracking_p2_1_2.yaml.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Tracking speed used to select desired belt-speed trial, e.g. 1.6.",
    )
    parser.add_argument(
        "--save-summary",
        default=None,
        help="Optional path for saving the summary figure as a PNG or PDF.",
    )
    parser.add_argument(
        "--controller-ap-filter-cutoff",
        type=float,
        default=AP_GRF_FILTER_CUTOFF_HZ,
        help="Cutoff used to display the filtered AP controller input.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the treadmill-speed and GRF summary plots.",
    )
    parser.add_argument(
        "--no-stickfigure",
        action="store_true",
        help="Skip the stickfigure visualization.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not keep matplotlib windows open; useful when only saving figures.",
    )
    parser.add_argument(
        "--reuse-model-cache",
        action="store_true",
        help="Reuse an existing biosym model cache instead of rebuilding it for this machine.",
    )
    parser.add_argument(
        "--plot-markers",
        action="store_true",
        help="Show biosym site markers in the stickfigure.",
    )
    parser.add_argument(
        "--plot-expected",
        action="store_true",
        help="Show expected markers if available in the objective metadata.",
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.2,
        help="Padding passed to the stickfigure plot for axis limits.",
    )
    args = parser.parse_args()

    payload = load_solution(args.result)
    states_dict, globals_dict, info, settings = extract_solution(payload)
    participant = resolve_participant(args)
    saved_model_file = settings.get("settings", {}).get("model")
    biosym_model = load_model_with_treadmill_contact(
        force_rebuild=not args.reuse_model_cache,
        model_file=saved_model_file,
    )

    print(f"Loaded result: {args.result}")
    print(f"Nodes: {len(states_dict)}")
    print(f"Duration: {float(globals_dict.dur):.6f} s")
    print(f"Speed: {float(globals_dict.speed):.6f} m/s")
    if isinstance(info, dict) and "status" in info:
        print(f"Solver status: {info['status']}")
    if isinstance(settings, dict):
        output_file = settings.get("settings", {}).get("output", {}).get("file")
        if output_file:
            print(f"Configured output file: {output_file}")

    tracking_speed = resolve_tracking_speed(args, globals_dict, settings)
    desired_grf_path = resolve_desired_grf_path(args, tracking_speed, participant)
    desired_grf = load_desired_grf(desired_grf_path, len(states_dict))
    desired_belt_speed = load_desired_belt_speed(args.belt_speeds, participant, tracking_speed, len(states_dict))
    joint_angle_reference_path = resolve_joint_angle_reference_path(settings, tracking_speed)
    desired_joint_angles = load_desired_joint_angles(joint_angle_reference_path, len(states_dict))
    experimental_duration = None
    if args.belt_speeds is not None and participant is not None:
        try:
            experimental_duration = get_duration(args.belt_speeds, participant, tracking_speed)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Experimental duration could not be loaded: {exc}")
    treadmill_controller_metadata = get_treadmill_controller_metadata(settings)
    if desired_grf_path is not None:
        print(f"Desired GRF source: {desired_grf_path}")
    if desired_grf_path is not None and desired_grf is None:
        print("Desired GRF could not be loaded; continuing without overlay.")
    if desired_belt_speed is not None:
        print(f"Desired belt-speed source: {args.belt_speeds} (participant {participant}, speed {tracking_speed})")
    elif participant is None and args.belt_speeds is not None:
        print("Desired belt speed could not infer participant; continuing without overlay.")
    if joint_angle_reference_path is not None:
        print(f"Experimental joint-angle source: {joint_angle_reference_path}")
    foot_strike_diagnosis = diagnose_foot_strike(
        biosym_model, states_dict, duration=float(globals_dict.dur)
    )
    print(foot_strike_label(foot_strike_diagnosis))

    summary_fig = None
    if not args.no_summary:
        summary_fig = create_summary_figure(
            biosym_model,
            states_dict,
            globals_dict,
            desired_grf=desired_grf,
            desired_belt_speed=desired_belt_speed,
            experimental_duration=experimental_duration,
            treadmill_controller_metadata=treadmill_controller_metadata,
            controller_ap_filter_cutoff_hz=args.controller_ap_filter_cutoff,
            foot_strike_diagnosis=foot_strike_diagnosis,
        )
        if args.save_summary:
            save_figure(summary_fig, args.save_summary)
            print(f"Saved summary figure to {args.save_summary}")
            joint_angle_path = joint_angle_output_path(args.save_summary)
            joint_angle_fig = create_joint_angle_figure(
                biosym_model,
                states_dict,
                desired_joint_angles=desired_joint_angles,
                foot_strike_diagnosis=foot_strike_diagnosis,
            )
            save_figure(joint_angle_fig, joint_angle_path)
            plt.close(joint_angle_fig)
            print(f"Saved joint-angle figure to {joint_angle_path}")
            slip_path = slip_friction_output_path(args.save_summary)
            slip_fig = create_slip_friction_figure(
                biosym_model, states_dict, duration=float(globals_dict.dur)
            )
            save_figure(slip_fig, slip_path)
            plt.close(slip_fig)
            print(f"Saved slip/friction figure to {slip_path}")
            diagnostics_path = contact_diagnostics_output_path(args.result)
            save_contact_diagnostics_csv(
                biosym_model,
                states_dict,
                diagnostics_path,
                duration=float(globals_dict.dur),
            )
            print(f"Saved contact diagnostics table to {diagnostics_path}")

    if args.no_stickfigure:
        if summary_fig is not None and not args.no_show:
            plt.show()
        return

    biosym_model.gc_model.visualization_duration = float(globals_dict.dur)

    stickfigure.plot_stick_figure(
        biosym_model,
        (states_dict, globals_dict),
        plot_markers=args.plot_markers,
        plot_expected=args.plot_expected,
        pad_ratio=args.pad_ratio,
    )


if __name__ == "__main__":
    main()
