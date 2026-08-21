from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import tracking_pipeline_common as base


ROOT = base.ROOT
RUN_OVERGROUND = ROOT / "bash_scripts" / "run_tracking_overground.sh"
TRACKING_CONFIG = "walking_tracking_overground.yaml"
DEFAULT_RESULT_LABEL = "overground_winter"


def contact_condition(args, participant=1):
    del args
    participant = int(participant)
    return {
        "model": (
            f"models/gait2d_scaled_p{participant}/"
            "gait2d_scaled_huntcrossley_6spheres.yaml"
        ),
        "result_dir": "huntcrossley_6spheres",
    }


def result_subject(args, participant):
    return f"p{participant}"


def parse_args():
    huntcrossley_parser = argparse.ArgumentParser(add_help=False)
    huntcrossley_parser.add_argument(
        "--initial-guess-file",
        default=None,
        help=(
            "Optional initial guess overriding the selected Hunt-Crossley "
            "standing solution."
        ),
    )
    huntcrossley_parser.add_argument(
        "--live-dashboard",
        action="store_true",
        help="Show the live Biosym/IPOPT convergence dashboard while solving.",
    )
    huntcrossley_parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8050,
        help="Local port for the live dashboard (default: 8050).",
    )
    huntcrossley_parser.add_argument(
        "--iteration-log-interval",
        type=int,
        default=5,
        help="Update the dashboard every N IPOPT iterations (default: 5).",
    )
    huntcrossley_parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build and evaluate the standing-based walking seed without running IPOPT.",
    )
    huntcrossley_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing result and its generated plots.",
    )
    huntcrossley_args, remaining = huntcrossley_parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = original_argv
    args.initial_guess_file = huntcrossley_args.initial_guess_file
    args.live_dashboard = huntcrossley_args.live_dashboard
    args.dashboard_port = huntcrossley_args.dashboard_port
    args.iteration_log_interval = huntcrossley_args.iteration_log_interval
    args.validate_only = huntcrossley_args.validate_only
    args.overwrite = huntcrossley_args.overwrite
    return args


def result_root(args, participant: int) -> Path:
    result_dir_name = contact_condition(args, participant)["result_dir"]
    participant_dir = ROOT / "result_HC" / result_subject(args, participant) / result_dir_name
    if args.no_node_directory:
        return participant_dir
    return participant_dir / f"{args.n_nodes}nodes"


def run_case(args, participant: int, speed: float):
    speed_label = base.speed_label(speed)
    result_dir = result_root(args, participant) / (
        args.result_label or DEFAULT_RESULT_LABEL
    )
    condition = contact_condition(args, participant)
    initial_guess = (
        ROOT / args.initial_guess_file
        if args.initial_guess_file is not None
        else ROOT / "result_HC" / result_subject(args, participant) / condition["result_dir"] / "standing.pkl"
    )
    output_file = result_dir / f"overground_{speed_label}.pkl"

    if output_file.exists():
        if args.skip_existing:
            print(
                f"\nSkipping Hunt-Crossley overground p{participant} "
                f"speed {speed:.1f}: {output_file.relative_to(ROOT)} exists"
            )
            return
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing result: {output_file}. "
                "Use a different --result-label, pass --skip-existing, "
                "or explicitly pass --overwrite."
            )
        print(f"\nOverwriting existing result: {output_file.relative_to(ROOT)}")
    if not initial_guess.exists() and not args.dry_run:
        raise FileNotFoundError(f"Initial guess does not exist: {initial_guess}")

    if not args.dry_run:
        result_dir.mkdir(parents=True, exist_ok=True)
    command = base.build_base_command(
        args,
        participant=participant,
        speed=speed,
        initial_guess=initial_guess.relative_to(ROOT),
        output_file=output_file.relative_to(ROOT),
        summary_file=(
            base.summary_file_for_result(args, output_file)
            if args.save_final_summary
            else None
        ),
        summary_from_memory=False,
    )
    command[0] = str(RUN_OVERGROUND)
    if "--tracking-config" not in command:
        command.extend(["--tracking-config", TRACKING_CONFIG])
    command.extend(
        [
            "--model-file", condition["model"],
        ]
    )
    if args.live_dashboard:
        command.extend(
            [
                "--live-dashboard",
                "--dashboard-port", str(args.dashboard_port),
                "--iteration-log-interval", str(args.iteration_log_interval),
            ]
        )
    if args.validate_only:
        command.append("--validate-only")

    print(
        f"\n=== Hunt-Crossley six-sphere overground "
        f"p{participant} speed {speed:.1f} ==="
    )
    base.run_output_command(
        command,
        dry_run=args.dry_run,
        output_file=output_file,
        label="Hunt-Crossley overground simulation",
    )


def main():
    args = parse_args()
    if any(participant < 1 or participant > 15 for participant in args.participants):
        raise ValueError("Hunt-Crossley participant models are available for P1-P15.")
    if args.real_only:
        raise ValueError("--real-only is a treadmill-pipeline option.")
    args.result_label = args.result_label or DEFAULT_RESULT_LABEL

    for participant in args.participants:
        for speed in args.speeds:
            run_case(copy.copy(args), participant, speed)


if __name__ == "__main__":
    main()
