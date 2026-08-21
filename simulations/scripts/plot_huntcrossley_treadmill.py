"""Plot a saved Hunt-Crossley treadmill stage without rerunning IPOPT."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cloudpickle
import matplotlib.pyplot as plt

from biosym.model import model

from src.contact_huntcrossley import install as install_huntcrossley_contact
from src.gait_data import DEFAULT_GAIT_DATA_FILE, get_duration
from src.gc_model_huntcrossley_treadmill import (
    HuntCrossleyTreadmill,
    MultiHuntCrossleyTreadmill,
)
from visualize_walking2d import (
    create_joint_angle_figure,
    create_slip_friction_figure,
    contact_diagnostics_output_path,
    create_summary_figure,
    diagnose_foot_strike,
    get_treadmill_controller_metadata,
    joint_angle_output_path,
    load_desired_belt_speed,
    load_desired_grf,
    load_desired_joint_angles,
    resolve_joint_angle_reference_path,
    save_figure,
    save_contact_diagnostics_csv,
    slip_friction_output_path,
)


def main() -> None:
    install_huntcrossley_contact()
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--participant", type=int, default=1)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--desired-grf", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result_path = Path(args.result)
    with result_path.open("rb") as handle:
        ((states, globals_), _info, settings) = cloudpickle.load(handle)

    model_path = settings["settings"]["model"]
    biosym_model = model.load_model(model_path, force_rebuild=True)
    if biosym_model.gc_model.__class__.__name__ == "MultiContact":
        biosym_model.gc_model = MultiHuntCrossleyTreadmill(biosym_model.gc_model)
    else:
        biosym_model.gc_model = HuntCrossleyTreadmill(biosym_model.gc_model)
    biosym_model.gc_model.process_eom(biosym_model)
    biosym_model._register_contact_model(biosym_model.gc_model)

    summary_path = Path(args.output) if args.output else result_path.with_suffix(".png")
    desired_grf = load_desired_grf(args.desired_grf, len(states))
    desired_belt = load_desired_belt_speed(
        str(DEFAULT_GAIT_DATA_FILE), args.participant, args.speed, len(states)
    )
    angle_path = resolve_joint_angle_reference_path(settings, args.speed)
    desired_angles = load_desired_joint_angles(angle_path, len(states))
    strike = diagnose_foot_strike(biosym_model, states)
    try:
        experimental_duration = get_duration(
            DEFAULT_GAIT_DATA_FILE, args.participant, args.speed
        )
    except (FileNotFoundError, ValueError):
        experimental_duration = None

    summary = create_summary_figure(
        biosym_model,
        states,
        globals_,
        desired_grf=desired_grf,
        desired_belt_speed=desired_belt,
        experimental_duration=experimental_duration,
        treadmill_controller_metadata=get_treadmill_controller_metadata(settings),
        foot_strike_diagnosis=strike,
    )
    save_figure(summary, summary_path)
    plt.close(summary)

    joint_path = joint_angle_output_path(summary_path)
    joints = create_joint_angle_figure(
        biosym_model,
        states,
        desired_joint_angles=desired_angles,
        foot_strike_diagnosis=strike,
    )
    save_figure(joints, joint_path)
    plt.close(joints)
    slip_path = slip_friction_output_path(summary_path)
    slip_figure = create_slip_friction_figure(biosym_model, states)
    save_figure(slip_figure, slip_path)
    plt.close(slip_figure)
    print(f"Saved belt-speed/GRF plot to {summary_path}")
    print(f"Saved joint-angle/moment plot to {joint_path}")
    print(f"Saved slip/friction plot to {slip_path}")
    diagnostics_path = contact_diagnostics_output_path(result_path)
    save_contact_diagnostics_csv(biosym_model, states, diagnostics_path)
    print(f"Saved contact diagnostics table to {diagnostics_path}")


if __name__ == "__main__":
    main()
