from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biosym.model import model
from src.metabolic_umberger import compute_umberger_metabolic_rate


RESULT_RE = re.compile(
    r"treadmill_(fixed_stage2|fixed|real|ideal|realistic)_(\d+)_(\d+)\.pkl$"
)

MODEL_LABELS = {
    "fixed_stage2": "ideal",
    "fixed": "ideal",
    "real": "real",
    "ideal": "ideal",
    "realistic": "real",
}

SIDES = ("r", "l")
JOINTS = ("hip", "knee", "ankle")
ACTIVATION_MUSCLES = ("gastroc", "soleus", "tibialis_ant")
ORIGINAL_MODEL_HEIGHT_M = 1.8


def participant_label(participant: str) -> str:
    value = participant.strip()
    return value.upper() if value.upper().startswith("P") else f"P{int(value)}"


def speed_from_match(match: re.Match[str]) -> float:
    return float(f"{match.group(2)}.{match.group(3)}")


def parse_speed_values(values: list[str] | None) -> set[float] | None:
    if values is None:
        return None
    return {round(float(value), 3) for value in values}


def parse_participants(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    return {participant_label(value) for value in values}


def resample_to_points(values: np.ndarray, n_points: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == n_points:
        return values.copy()
    old_x = np.linspace(0.0, 100.0, values.size)
    new_x = np.linspace(0.0, 100.0, n_points)
    return np.interp(new_x, old_x, values)


def get_column(array: np.ndarray, names: list[str], name: str) -> np.ndarray:
    return np.asarray(array, dtype=float)[:, names.index(name)]


def load_result(path: Path, participant_model):
    with path.open("rb") as handle:
        payload = cloudpickle.load(handle)
    trajectory_or_states, globals_dict = payload[0]
    if hasattr(trajectory_or_states, "states"):
        trajectory = trajectory_or_states
    else:
        trajectory = SimpleNamespace(
            states=trajectory_or_states,
            constants=participant_model.default_constants,
        )
    return trajectory, globals_dict


def discover_results(result_root: Path, result_subdir: str | None = None) -> list[Path]:
    paths = result_root.glob("p*/**/treadmill_*.pkl")
    if result_subdir:
        paths = (path for path in paths if path.parent.name == result_subdir)
    return sorted(paths)


def participant_from_result_path(path: Path, result_root: Path) -> str:
    try:
        return path.relative_to(result_root).parts[0]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Could not infer participant from result path: {path}") from error


def default_output_path(result_subdir: str | None, n_points: int) -> Path:
    suffix = ""
    if result_subdir:
        subdir_label = re.sub(r"[^A-Za-z0-9]+", "_", result_subdir).strip("_")
        suffix = f"_{subdir_label}"
    return Path("result") / f"simulation_results_{n_points}_points{suffix}.csv"


class ModelCache:
    def __init__(self, root: Path):
        self.root = root
        self._cache = {}
        self._height_cache = {}
        self._body_mass_cache = {}

    def load(self, participant: str, result_path: Path):
        model_dir = self.root / "models" / f"gait2d_scaled_{participant}"
        relative_parts = result_path.relative_to(self.root).parts
        variant_names = [part for part in relative_parts if part.startswith("huntcrossley_")]
        model_path = model_dir / "gait2d_scaled.yaml"
        if variant_names:
            variant_path = model_dir / f"gait2d_scaled_{variant_names[-1]}.yaml"
            if variant_path.is_file():
                model_path = variant_path
        if not model_path.is_file():
            available = sorted(model_dir.glob("gait2d_scaled*.yaml"))
            if len(available) == 1:
                model_path = available[0]
            else:
                raise FileNotFoundError(
                    f"Could not select a model YAML for {participant} and {result_path}; "
                    f"available files: {available}"
                )

        cache_key = (participant, model_path.name)
        if cache_key not in self._cache:
            self._cache[cache_key] = model.load_model(str(model_path), force_rebuild=False)
        return self._cache[cache_key]

    def height(self, participant: str) -> float:
        if participant not in self._height_cache:
            original_xml = self.root / "models" / "gait2d" / "gait2d.xml"
            scaled_xml = self.root / "models" / f"gait2d_scaled_{participant}" / "gait2d_scaled.xml"
            original_site_size = float(ET.parse(original_xml).find(".//default/site").get("size").split()[0])
            scaled_site_size = float(ET.parse(scaled_xml).find(".//default/site").get("size").split()[0])
            self._height_cache[participant] = ORIGINAL_MODEL_HEIGHT_M * scaled_site_size / original_site_size
        return self._height_cache[participant]

    def body_mass(self, participant: str) -> float:
        if participant not in self._body_mass_cache:
            scaled_xml = self.root / "models" / f"gait2d_scaled_{participant}" / "gait2d_scaled.xml"
            self._body_mass_cache[participant] = sum(
                float(inertial.get("mass"))
                for inertial in ET.parse(scaled_xml).findall(".//inertial")
            )
        return self._body_mass_cache[participant]


def trial_metadata(
    *,
    participant: str,
    treadmill_model: str,
    speed_m_per_s: float,
    duration_s: float,
    body_mass_kg: float,
    height_m: float,
    metabolic_rate_w_per_kg: float,
) -> dict[str, float | str]:
    stride_length_m = speed_m_per_s * duration_s
    step_length_m = stride_length_m / 2.0
    return {
        "participant": participant_label(participant),
        "treadmill_model": MODEL_LABELS[treadmill_model],
        "speed": speed_m_per_s,
        "duration": duration_s,
        "duration_s": duration_s,
        "stride_length_m": stride_length_m,
        "step_length_m": step_length_m,
        "cadence_steps_per_min": 120.0 / duration_s,
        "body_mass_kg": body_mass_kg,
        "height_m": height_m,
        "bmi_kg_per_m2": body_mass_kg / height_m**2,
        "metabolic_rate_w_per_kg": metabolic_rate_w_per_kg,
    }


def grf_column_name(ext_force_name: str) -> str:
    _, _, side, axis = ext_force_name.split("_")
    return f"GRF_{side}_{axis}"


def muscle_name(muscle: dict) -> str:
    return str(muscle.get("name", muscle.get("Name")))


def property_names(properties) -> list[str]:
    names = getattr(properties, "names", None)
    if names is not None:
        return list(names)
    return list(properties["names"])


def build_rows_for_result(
    *,
    path: Path,
    participant: str,
    participant_model,
    body_mass_kg: float,
    height_m: float,
    n_points: int,
    metabolic_formulation: str,
) -> list[dict]:
    match = RESULT_RE.match(path.name)
    if match is None:
        return []

    treadmill_model = match.group(1)
    speed_m_per_s = speed_from_match(match)

    trajectory, globals_dict = load_result(path, participant_model)
    duration_s = float(globals_dict.dur)
    body_weight_n = body_mass_kg * 9.81

    named_state_format = not hasattr(trajectory.states, "model")
    if named_state_format:
        states_model = np.asarray(trajectory.states.q, dtype=float)
    else:
        states_model = np.asarray(trajectory.states.model, dtype=float)
    states_gc = np.asarray(trajectory.states.gc_model, dtype=float)
    states_actuator = np.asarray(trajectory.states.actuator_model, dtype=float)
    total_moments = np.asarray(
        participant_model.run["actuator_model"](trajectory.states, trajectory.constants),
        dtype=float,
    )

    metabolic = compute_umberger_metabolic_rate(
        participant_model,
        trajectory,
        duration=duration_s,
        body_mass=body_mass_kg,
        formulation=metabolic_formulation,
    )

    coordinate_names = property_names(participant_model.coordinates)
    force_properties = getattr(participant_model, "forces", participant_model.tau)
    force_names = property_names(force_properties)
    force_names = [f"M_{name[2:]}" if name.startswith("t_") else name for name in force_names]
    ext_force_names = property_names(participant_model.ext_forces)
    actuator = participant_model.actuator_model
    activation_indices = np.asarray(actuator.idx["a"], dtype=int)
    actuator_muscle_names = [muscle_name(muscle) for muscle in actuator.muscles_dict]

    q_by_name = {
        name: resample_to_points(get_column(states_model, coordinate_names, name), n_points)
        for name in coordinate_names
    }
    moment_by_name = {
        name: resample_to_points(get_column(total_moments, force_names, name) / body_mass_kg, n_points)
        for name in force_names
    }
    moment_nm_by_name = {
        name: resample_to_points(get_column(total_moments, force_names, name), n_points)
        for name in force_names
    }
    if named_state_format:
        contact_forces = []
        for node in range(len(trajectory.states.q)):
            complete_states = participant_model.default_states.replace(
                q=trajectory.states.q[node],
                qd=trajectory.states.qd[node],
                gc_model=trajectory.states.gc_model[node],
                actuator_model=trajectory.states.actuator_model[node],
            )
            forces, _ = participant_model.run["gc_model"](
                complete_states, trajectory.constants
            )
            contact_forces.append(np.asarray(forces, dtype=float).reshape(-1))
        contact_forces = np.asarray(contact_forces)
        ext_force_by_name = {
            name: resample_to_points(contact_forces[:, i], n_points)
            for i, name in enumerate(ext_force_names)
        }
    else:
        ext_force_start = int(participant_model.ext_forces["idx"])
        ext_force_by_name = {
            name: resample_to_points(states_model[:, ext_force_start + i], n_points)
            for i, name in enumerate(ext_force_names)
        }
    activation_by_name = {
        name: resample_to_points(states_actuator[:, activation_indices[actuator_muscle_names.index(name)]], n_points)
        for side in SIDES
        for muscle in ACTIVATION_MUSCLES
        for name in [f"{side}_{muscle}"]
    }
    belt_speed_r = resample_to_points(states_gc[:, 0], n_points)
    belt_speed_l = resample_to_points(states_gc[:, 1], n_points)
    node_metabolic_w = resample_to_points(metabolic.total, n_points)

    base = trial_metadata(
        participant=participant,
        treadmill_model=treadmill_model,
        speed_m_per_s=speed_m_per_s,
        duration_s=duration_s,
        body_mass_kg=body_mass_kg,
        height_m=height_m,
        metabolic_rate_w_per_kg=float(metabolic.metabolic_rate_w_per_kg),
    )

    rows = []
    percent_gait = np.linspace(0.0, 100.0, n_points)
    for node in range(n_points):
        row = dict(base)
        row["node"] = node
        row["percent_gait"] = float(percent_gait[node])
        row["time_s"] = float(duration_s * percent_gait[node] / 100.0)
        row["node_metabolic_rate_w_per_kg"] = float(node_metabolic_w[node] / body_mass_kg)
        row["belt_speed_r"] = float(belt_speed_r[node])
        row["belt_speed_l"] = float(belt_speed_l[node])
        row["belt_speed_r_abs"] = float(abs(belt_speed_r[node]))
        row["belt_speed_l_abs"] = float(abs(belt_speed_l[node]))

        row["q_pelvis_tx_m"] = float(q_by_name["q_pelvis_tx"][node])
        row["q_pelvis_ty_m"] = float(q_by_name["q_pelvis_ty"][node])
        row["q_pelvis_tilt_deg"] = float(np.rad2deg(q_by_name["q_pelvis_tilt"][node]))

        for side in SIDES:
            for joint in JOINTS:
                q_name = f"q_{joint}_{side}"
                m_name = f"M_{joint}_{side}"
                row[f"{q_name}_deg"] = float(np.rad2deg(q_by_name[q_name][node]))
                row[f"{m_name}_nm"] = float(moment_nm_by_name[m_name][node])
                row[f"{m_name}_nm_per_kg"] = float(moment_by_name[m_name][node])

        for name, values in ext_force_by_name.items():
            row[grf_column_name(name)] = float(values[node] / body_weight_n)

        for side in SIDES:
            for muscle in ACTIVATION_MUSCLES:
                name = f"{side}_{muscle}"
                row[f"activation_{muscle}_{side}"] = float(activation_by_name[name][node])

        rows.append(row)

    return rows


def export_results(args: argparse.Namespace) -> pd.DataFrame:
    result_root = Path(args.result_root)
    if not result_root.is_absolute():
        result_root = ROOT / result_root
    output = Path(args.output) if args.output else default_output_path(args.result_subdir, args.n_points)
    if not output.is_absolute():
        output = ROOT / output

    participants = parse_participants(args.participants)
    speeds = parse_speed_values(args.speeds)
    treadmill_models = set(args.treadmill_models) if args.treadmill_models else None

    model_cache = ModelCache(ROOT)
    rows = []
    result_paths = discover_results(result_root, args.result_subdir)
    if args.paired_only:
        models_by_case: dict[tuple[str, float], set[str]] = {}
        for path in result_paths:
            match = RESULT_RE.match(path.name)
            if match is None:
                continue
            participant = participant_label(participant_from_result_path(path, result_root))
            case = (participant, round(speed_from_match(match), 3))
            models_by_case.setdefault(case, set()).add(match.group(1))
        paired_cases = {
            case
            for case, models in models_by_case.items()
            if (
                {"fixed_stage2", "real"}.issubset(models)
                or {"fixed", "real"}.issubset(models)
                or {"ideal", "realistic"}.issubset(models)
            )
        }
        result_paths = [
            path
            for path in result_paths
            if (match := RESULT_RE.match(path.name)) is not None
            and (
                participant_label(participant_from_result_path(path, result_root)),
                round(speed_from_match(match), 3),
            )
            in paired_cases
        ]
    for path in result_paths:
        match = RESULT_RE.match(path.name)
        if match is None:
            continue
        participant = participant_from_result_path(path, result_root)
        if participants is not None and participant_label(participant) not in participants:
            continue
        treadmill_model = match.group(1)
        if treadmill_models is not None and treadmill_model not in treadmill_models:
            continue
        speed = round(speed_from_match(match), 3)
        if speeds is not None and speed not in speeds:
            continue

        participant_model = model_cache.load(participant, path)
        body_mass_kg = model_cache.body_mass(participant)
        height_m = model_cache.height(participant)
        rows.extend(
            build_rows_for_result(
                path=path,
                participant=participant,
                participant_model=participant_model,
                body_mass_kg=body_mass_kg,
                height_m=height_m,
                n_points=args.n_points,
                metabolic_formulation=args.metabolic_formulation,
            )
        )

    data = pd.DataFrame(rows)
    if data.empty:
        raise RuntimeError("No result files matched the requested filters.")

    data = data.sort_values(["participant", "speed", "treadmill_model", "node"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    try:
        output_label = output.relative_to(ROOT)
    except ValueError:
        output_label = output
    print(f"Saved {len(data)} rows x {len(data.columns)} columns to {output_label}")
    print(f"Trials exported: {data[['participant', 'speed', 'treadmill_model']].drop_duplicates().shape[0]}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all simulation result trajectories to one plotting-friendly CSV."
    )
    parser.add_argument("--result-root", default="result", help="Root result directory.")
    parser.add_argument(
        "--result-subdir",
        default=None,
        help=(
            "Optional subdirectory under each result/p*/ participant folder, "
            "for example '100nodes' or '100nodes/no_duration'. "
            "By default, scan the participant folder itself."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. By default, use result/simulation_results_<n>_points.csv; "
            "when --result-subdir is set, append the subdirectory name to avoid overwriting it."
        ),
    )
    parser.add_argument("--n-points", type=int, default=101, help="Number of gait-cycle points per trial.")
    parser.add_argument(
        "--metabolic-formulation",
        choices=["original", "continuous"],
        default="original",
        help="Umberger formulation for metabolic-rate export.",
    )
    parser.add_argument("--participants", nargs="*", help="Optional participant filters, e.g. p1 p2.")
    parser.add_argument("--speeds", nargs="*", help="Optional speed filters, e.g. 1.0 1.4.")
    parser.add_argument(
        "--treadmill-models",
        nargs="*",
        choices=["fixed_stage2", "fixed", "real", "ideal", "realistic"],
        help="Optional treadmill model filters.",
    )
    parser.add_argument(
        "--paired-only",
        action="store_true",
        help=(
            "Export a participant-speed case only when both ideal and realistic "
            "result files are available (fixed/real and legacy fixed_stage2/real names also work)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    export_results(parse_args())


if __name__ == "__main__":
    main()
