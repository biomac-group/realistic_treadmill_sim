from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import tracking_pipeline_common as base
from run_tracking_pipeline_huntcrossley_overground import (
    contact_condition,
    result_subject,
)


ROOT = base.ROOT
RUN_TRACKING = ROOT / "bash_scripts" / "run_tracking.sh"
TRACKING_CONFIG = "walking_tracking_template.yaml"


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--ideal-only",
        action="store_true",
        help="Solve only the fixed-speed (ideal treadmill) stage for every case.",
    )
    parser.add_argument(
        "--no-post-hs-interpolation",
        action="store_true",
        help="Use the raw AP contact force in the realistic treadmill controller.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--live-dashboard", action="store_true")
    parser.add_argument("--dashboard-port", type=int, default=8050)
    parser.add_argument("--iteration-log-interval", type=int, default=5)
    parser.add_argument(
        "--transform-overground-frame",
        action="store_true",
        help=(
            "For the combined overground-to-treadmill pipeline, subtract "
            "pelvis progression and initialize both belts at the target speed."
        ),
    )
    parser.add_argument(
        "--initial-guess-file",
        type=Path,
        default=None,
        help=(
            "Optional result PKL used as the fixed/ideal treadmill initial guess. "
            "Defaults to the participant's Hunt-Crossley standing result."
        ),
    )
    parser.add_argument(
        "--initial-guess-label",
        default=None,
        help=(
            "Optional prior result subdirectory used for per-speed ideal warm starts. "
            "For each case, load treadmill_fixed_<speed>.pkl from this label."
        ),
    )
    special, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = original
    args.overwrite = special.overwrite
    args.ideal_only = special.ideal_only
    args.no_post_hs_interpolation = special.no_post_hs_interpolation
    args.validate_only = special.validate_only
    args.live_dashboard = special.live_dashboard
    args.dashboard_port = special.dashboard_port
    args.iteration_log_interval = special.iteration_log_interval
    args.transform_overground_frame = special.transform_overground_frame
    args.initial_guess_file = special.initial_guess_file
    args.initial_guess_label = special.initial_guess_label
    if args.initial_guess_file is not None and args.initial_guess_label is not None:
        parser.error("Use either --initial-guess-file or --initial-guess-label, not both.")
    if args.real_only and args.ideal_only:
        parser.error("Use either --real-only or --ideal-only, not both.")
    if args.real_only and args.initial_guess_file is None and args.initial_guess_label is None:
        parser.error(
            "--real-only requires --initial-guess-file or --initial-guess-label."
        )
    return args


def result_root(args, participant):
    condition = contact_condition(args, participant)
    root = ROOT / "result_HC" / result_subject(args, participant) / condition["result_dir"]
    return root if args.no_node_directory else root / f"{args.n_nodes}nodes"


def run_stage(args, participant, speed, name, initial_guess, output, zero_controller, controller_edge_source=None):
    if output.exists():
        if args.skip_existing:
            print(f"Skipping {name}: {output.relative_to(ROOT)} exists")
            return
        if not args.overwrite:
            raise FileExistsError(
                f"{output} exists; pass --overwrite or --skip-existing."
            )
        print(f"Overwriting {output.relative_to(ROOT)}")
    if not initial_guess.exists() and not args.dry_run:
        raise FileNotFoundError(f"Missing initial guess: {initial_guess}")

    condition = contact_condition(args, participant)
    command = base.build_base_command(
        args,
        participant=participant,
        speed=speed,
        initial_guess=initial_guess.relative_to(ROOT),
        output_file=output.relative_to(ROOT),
        summary_file=output.with_suffix(".png"),
    )
    command[0] = str(RUN_TRACKING)
    if "--tracking-config" not in command:
        command.extend(["--tracking-config", TRACKING_CONFIG])
    command.extend(
        [
            "--model-file", condition["model"],
        ]
    )
    if zero_controller:
        command.extend(["--k-fy", "0", "--k-fx", "0", "--k-p", "0", "--k-d", "0"])
    # The fixed/ideal controller gains are all zero, so the post-HS AP-force
    # bridge cannot affect this stage. Avoid building it unnecessarily.
    if zero_controller or args.no_post_hs_interpolation:
        command.append("--no-post-hs-interpolation")
    if controller_edge_source is not None and not args.no_post_hs_interpolation:
        source_flag = (
            "--controller-edge-source-file"
            if Path(controller_edge_source).suffix.lower() == ".csv"
            else "--controller-edge-source-result"
        )
        command.extend([source_flag, str(controller_edge_source)])
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
    print(f"\n=== Hunt-Crossley treadmill {name} | p{participant} {speed:.1f} m/s ===")
    base.run_output_command(
        command,
        dry_run=args.dry_run,
        output_file=output,
        label=f"Hunt-Crossley treadmill {name}",
    )


def run_case(args, participant, speed):
    condition = contact_condition(args, participant)
    case_label = args.result_label or "treadmill_winter"
    case_root = result_root(args, participant) / case_label
    label = base.speed_label(speed)
    standing = (
        ROOT / "result_HC" / result_subject(args, participant) / condition["result_dir"] / "standing.pkl"
    )
    if args.initial_guess_label is not None:
        fixed_initial_guess = (
            result_root(args, participant)
            / args.initial_guess_label
            / f"treadmill_fixed_{label}.pkl"
        )
    elif args.initial_guess_file is not None:
        fixed_initial_guess = (
            args.initial_guess_file
            if args.initial_guess_file.is_absolute()
            else ROOT / args.initial_guess_file
        )
    else:
        fixed_initial_guess = standing
    fixed = case_root / f"treadmill_fixed_{label}.pkl"
    fixed_contact_diagnostics = fixed.with_name(
        f"{fixed.stem}_contact_diagnostics.csv"
    )
    real = case_root / f"treadmill_real_{label}.pkl"
    case_root.mkdir(parents=True, exist_ok=True)
    if args.real_only:
        # In real-only mode the explicitly selected guess is already the fixed
        # solution to warm-start from.  Prefer its adjacent contact diagnostics;
        # if they are absent, let script_tracking derive controller edges from
        # the result PKL itself.
        real_contact_diagnostics = fixed_initial_guess.with_name(
            f"{fixed_initial_guess.stem}_contact_diagnostics.csv"
        )
        controller_edge_source = (
            real_contact_diagnostics
            if real_contact_diagnostics.exists() or args.dry_run
            else fixed_initial_guess
        )
        run_stage(
            args,
            participant,
            speed,
            "real",
            fixed_initial_guess,
            real,
            False,
            controller_edge_source=controller_edge_source,
        )
        return

    run_stage(args, participant, speed, "fixed", fixed_initial_guess, fixed, True)
    if not args.validate_only and not args.ideal_only:
        run_stage(
            args,
            participant,
            speed,
            "real",
            fixed,
            real,
            False,
            controller_edge_source=fixed_contact_diagnostics,
        )


def main():
    args = parse_args()
    if any(participant < 1 or participant > 15 for participant in args.participants):
        raise ValueError("Hunt-Crossley participant models are available for P1-P15.")
    failed_cases = []
    for participant in args.participants:
        for speed in args.speeds:
            try:
                run_case(copy.copy(args), participant, speed)
            except Exception as exc:
                failed_cases.append((participant, speed, exc))
                print(
                    f"\nFAILED Hunt-Crossley treadmill case p{participant} "
                    f"{speed:.1f} m/s: {type(exc).__name__}: {exc}\n"
                    "Continuing with the next case."
                )

    if failed_cases:
        print("\nHunt-Crossley treadmill pipeline completed with failed cases:")
        for participant, speed, exc in failed_cases:
            print(
                f"  p{participant} {speed:.1f} m/s: "
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()
