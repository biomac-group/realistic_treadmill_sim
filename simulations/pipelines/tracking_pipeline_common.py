from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_TRACKING = ROOT / "bash_scripts" / "run_tracking.sh"


class IterationLimitExceeded(RuntimeError):
    """Raised when IPOPT stops because the configured iteration limit was hit."""


def speed_label(speed: float) -> str:
    return f"{abs(float(speed)):.1f}".replace(".", "_")


def output_exceeded_iteration_limit(output: str | None) -> bool:
    if not output:
        return False
    patterns = [
        "Maximum Number of Iterations Exceeded",
        "Maximum_Iterations_Exceeded",
        "maximum number of iterations exceeded",
    ]
    return any(pattern in output for pattern in patterns)


def run_command(command: list[str], dry_run: bool):
    print("\n" + " ".join(command))
    if dry_run:
        return ""

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        output_lines.append(line)
    returncode = process.wait()
    output = "".join(output_lines)
    if output_exceeded_iteration_limit(output):
        raise IterationLimitExceeded
    if returncode:
        raise subprocess.CalledProcessError(returncode, command, output=output)
    return output


def is_native_abort(returncode: int) -> bool:
    return returncode in {-signal.SIGABRT, 128 + signal.SIGABRT}


def run_output_command(command: list[str], dry_run: bool, output_file: Path, label: str):
    if not dry_run:
        snapshot_model_inputs(command, output_file)
    try:
        run_command(command, dry_run=dry_run)
    except subprocess.CalledProcessError as exc:
        if output_exceeded_iteration_limit(exc.output):
            raise IterationLimitExceeded from exc
        if is_native_abort(exc.returncode) and output_file.exists():
            print(
                f"\n{label} process aborted after writing the expected output; "
                f"continuing with {output_file.relative_to(ROOT)}"
            )
            return
        raise


def snapshot_model_inputs(command: list[str], output_file: Path) -> None:
    """Archive the exact model YAML and referenced XML inputs for a solve."""
    if "--model-file" not in command:
        return

    model_arg = Path(command[command.index("--model-file") + 1])
    model_file = model_arg if model_arg.is_absolute() else ROOT / model_arg
    if not model_file.exists():
        raise FileNotFoundError(f"Cannot snapshot missing model file: {model_file}")

    with model_file.open("r", encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)

    referenced_files = [("model_yaml", model_file)]
    model_section = (model_config or {}).get("model", {})
    model_name = model_section.get("name")
    if model_name:
        referenced_files.append(("model_xml", model_file.parent / model_name))
    parameters = model_section.get("additional_parameters", {})
    for key, value in parameters.items():
        if isinstance(value, dict) and value.get("file"):
            referenced_files.append((key, model_file.parent / value["file"]))

    output_file = output_file if output_file.is_absolute() else ROOT / output_file
    snapshot_dir = output_file.parent / "input_snapshots" / output_file.stem
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for role, source in referenced_files:
        source = source.resolve()
        if not source.exists():
            raise FileNotFoundError(f"Cannot snapshot missing {role}: {source}")
        destination = snapshot_dir / f"{role}{source.suffix}"
        shutil.copy2(source, destination)
        manifest[role] = {
            "source": str(source),
            "snapshot": destination.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }

    manifest_file = snapshot_dir / "manifest.json"
    with manifest_file.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Archived simulation inputs in {snapshot_dir.relative_to(ROOT)}")


def grf_file_for_case(
    participant: int,
    speed: float,
) -> Path:
    token = f"{abs(float(speed)):.1f}".replace(".", "p")
    return Path("data") / "opensim_exports" / f"p{participant}_grf_mean_speed_{token}.mot"


def summary_file_for_result(args, result_file: Path) -> Path:
    if args.summary_file is None:
        return result_file.with_suffix(".png")
    summary_file = Path(args.summary_file)
    if not summary_file.is_absolute() and summary_file.parent == Path("."):
        summary_file = result_file.with_name(summary_file.name)
    elif not summary_file.is_absolute():
        summary_file = ROOT / summary_file
    if summary_file.suffix == "":
        summary_file = summary_file.with_suffix(".png")
    return summary_file


def path_for_command(path: Path) -> str:
    if path.is_absolute():
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)
    return str(path)


def add_summary_args(command: list[str], summary_file: Path | None, from_memory: bool):
    if summary_file is None:
        command.append("--no-auto-summary")
    else:
        command.extend(["--summary-file", path_for_command(summary_file)])
        if from_memory:
            command.append("--summary-from-memory")


def build_base_command(
    args,
    participant: int,
    speed: float,
    initial_guess: Path,
    output_file: Path,
    summary_file: Path | None = None,
    summary_from_memory: bool = True,
) -> list[str]:
    command = [
        str(RUN_TRACKING),
        "--participant",
        str(participant),
        "--speed",
        f"{speed:.1f}",
        "--n-nodes",
        str(args.n_nodes),
        "--initial-guess-file",
        str(initial_guess),
        "--output-file",
        str(output_file),
    ]
    if args.tracking_config is not None:
        command.extend(["--tracking-config", args.tracking_config])
    command.extend(["--effort-model", args.effort_model])
    if args.muscle_effort_weight is not None:
        command.extend(["--muscle-effort-weight", str(args.muscle_effort_weight)])
    if args.grf_force_threshold is not None:
        command.extend(["--grf-force-threshold", str(args.grf_force_threshold)])
    if args.grf_weight is not None:
        command.extend(["--grf-weight", str(args.grf_weight)])
    command.append(
        "--muscle-effort-speedweighting"
        if args.muscle_effort_speedweighting
        else "--no-muscle-effort-speedweighting"
    )
    grf_reference = grf_file_for_case(participant, speed)
    if not (ROOT / grf_reference).exists() and not args.dry_run:
        raise FileNotFoundError(f"GRF reference was not found: {ROOT / grf_reference}")
    command.extend(["--grf-reference-file", str(grf_reference)])
    # Match the retained Winter cadence groups: 1.0--1.4 m/s use natural
    # cadence and faster trials use fast cadence.
    cadence = "natural" if abs(float(speed)) <= 1.4 else "fast"
    angle_reference = (
        ROOT
        / "data"
        / "opensim_exports"
        / "winter_1987"
        / f"winter_1987_{cadence}_cadence_kinematics_mean_var_rad.mot"
    )
    if not angle_reference.exists() and not args.dry_run:
        raise FileNotFoundError(
            f"Winter 1987 {cadence}-cadence angle reference was not found: {angle_reference}"
        )
    command.extend(["--angle-reference-file", str(angle_reference.relative_to(ROOT))])
    if not args.visualize:
        command.append("--no-visualize")
    add_summary_args(command, summary_file, from_memory=summary_from_memory)
    return command


def parse_args():
    parser = argparse.ArgumentParser(description="Run the three-stage treadmill tracking pipeline.")
    parser.add_argument("--participants", type=int, nargs="+", required=True, help="Participant numbers, e.g. 2 3 4.")
    parser.add_argument("--speeds", type=float, nargs="+", required=True, help="Speeds, e.g. 1.2 1.4 1.6 1.8.")
    parser.add_argument("--n-nodes", type=int, default=100, help="Collocation node count.")
    parser.add_argument("--tracking-config", default=None, help="Optional tracking YAML template/config.")
    parser.add_argument(
        "--effort-model",
        choices=("activation", "muscle_volume"),
        default="activation",
        help=(
            "Use BioSym's built-in effort term with equal actuator weights "
            "or normalized muscle-volume weights."
        ),
    )
    parser.add_argument(
        "--muscle-effort-weight",
        type=float,
        default=None,
        help="Optional weight override for BioSym's effort objective.",
    )
    parser.add_argument(
        "--muscle-effort-speedweighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Divide effort by the fixed requested walking speed^exponent.",
    )
    parser.add_argument(
        "--grf-force-threshold",
        type=float,
        default=None,
        help=(
            "Optionally track only experimental GRF components whose own "
            "absolute magnitude exceeds this threshold in N."
        ),
    )
    parser.add_argument(
        "--grf-weight",
        type=float,
        default=None,
        help="Optional weight override for the track_grf objective.",
    )
    parser.add_argument(
        "--no-node-directory",
        action="store_true",
        help="Store results directly under result/p*/ instead of result/p*/<n>nodes/.",
    )
    parser.add_argument(
        "--result-label",
        default=None,
        help=(
            "Optional result subdirectory. By default it is under result/p*/<n>nodes; "
            "with --no-node-directory it is directly under result/p*/."
        ),
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Run only the final real/controller stage, using an existing fixed stage 2 as the initial guess.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip stages whose output file already exists.")
    parser.add_argument(
        "--summary-file",
        default=None,
        help=(
            "Custom summary figure filename/path. A bare filename is saved next to the result; "
            "a relative path is resolved from the repository root."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--no-save-final-summary",
        dest="save_final_summary",
        action="store_false",
        help="Do not save visualize_walking2d summary plots for stage results.",
    )
    parser.set_defaults(save_final_summary=True)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show the walking visualization after each solve. Off by default for batch runs.",
    )
    return parser.parse_args()
