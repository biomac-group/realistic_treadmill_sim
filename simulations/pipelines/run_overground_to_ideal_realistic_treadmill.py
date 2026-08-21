"""Solve overground gait, then branch to ideal and realistic treadmills."""

from __future__ import annotations

import copy
from pathlib import Path

import tracking_pipeline_common as base
import run_tracking_pipeline_huntcrossley_overground as overground
import run_tracking_pipeline_huntcrossley_treadmill as treadmill
from src.overground_to_treadmill_seed import convert_overground_result


ROOT = base.ROOT


def main():
    args = treadmill.parse_args()
    if any(participant < 1 or participant > 15 for participant in args.participants):
        raise ValueError("Hunt-Crossley participant models are available for P1-P15.")
    args.initial_guess_file = None
    args.result_label = args.result_label or "overground_winter"

    for participant in args.participants:
        for speed in args.speeds:
            case_args = copy.copy(args)
            overground.run_case(case_args, participant, speed)

            root = treadmill.result_root(args, participant)
            label = base.speed_label(speed)
            overground_result = (
                root / args.result_label / f"overground_{label}.pkl"
            )
            # Keep the overground result and both treadmill branches together
            # in the selected result-label directory.
            case_root = root / args.result_label
            ideal = case_root / f"treadmill_ideal_{label}.pkl"
            realistic = case_root / f"treadmill_realistic_{label}.pkl"

            if not args.dry_run:
                case_root.mkdir(parents=True, exist_ok=True)

            if not args.dry_run and not overground_result.exists():
                raise FileNotFoundError(
                    f"Overground branch input was not created: {overground_result}"
                )

            treadmill_initial_guess = overground_result
            if args.transform_overground_frame:
                treadmill_initial_guess = (
                    case_root / f"overground_treadmill_frame_seed_{label}.pkl"
                )
                if args.dry_run:
                    print(
                        f"Would convert {overground_result.relative_to(ROOT)} to "
                        f"{treadmill_initial_guess.relative_to(ROOT)}"
                    )
                elif treadmill_initial_guess.exists() and args.skip_existing:
                    print(
                        "Reusing converted treadmill-frame seed: "
                        f"{treadmill_initial_guess.relative_to(ROOT)}"
                    )
                else:
                    convert_overground_result(
                        overground_result,
                        treadmill_initial_guess,
                        speed,
                        overwrite=args.overwrite,
                    )

            # Both branches start from the same overground-derived gait;
            # realistic does not start from ideal.
            treadmill.run_stage(
                args,
                participant,
                speed,
                "ideal",
                treadmill_initial_guess,
                ideal,
                True,
            )
            if not args.validate_only:
                treadmill.run_stage(
                    args,
                    participant,
                    speed,
                    "realistic",
                    treadmill_initial_guess,
                    realistic,
                    False,
                    controller_edge_source=overground_result,
                )


if __name__ == "__main__":
    main()
