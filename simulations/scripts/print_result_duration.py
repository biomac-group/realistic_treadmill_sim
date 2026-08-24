import argparse
import re
from pathlib import Path

import cloudpickle

from src.gait_data import DEFAULT_GAIT_DATA_FILE, get_duration


def infer_duration_lookup(settings, result_path: Path) -> tuple[int | None, float | None]:
    for objective in settings.get("objectives", []):
        if objective.get("name") != "track_duration":
            continue
        args = objective.get("args", {})
        person = args.get("person", args.get("participant"))
        speed = args.get("speed")
        if person is not None and speed is not None:
            return int(person), round(abs(float(speed)), 1)

    person_match = re.search(r"p(\d+)", str(result_path))
    person = int(person_match.group(1)) if person_match else None

    speed_match = re.search(r"_(\d+)_(\d+)(?:[^0-9].*)?\.pkl$", result_path.name)
    speed = float(f"{speed_match.group(1)}.{speed_match.group(2)}") if speed_match else None
    if speed is not None:
        speed = round(abs(speed), 1)

    return person, speed


def load_real_duration(datafile: Path, person: int | None, speed: float | None) -> float | None:
    if person is None or speed is None:
        return None
    return get_duration(datafile, person, speed)


def main():
    parser = argparse.ArgumentParser(description="Print the simulated duration from a biosym result pickle.")
    parser.add_argument("result", help="Path to the result .pkl file.")
    parser.add_argument("--durations-file", default=str(DEFAULT_GAIT_DATA_FILE), help="Path to gait_data.csv.")
    parser.add_argument("--person", type=int, help="Override participant/person index.")
    parser.add_argument("--speed", type=float, help="Override walking speed in m/s.")
    args = parser.parse_args()

    with open(args.result, "rb") as handle:
        payload = cloudpickle.load(handle)

    (states_dict, globals_dict), info, settings = payload
    sim_duration = float(globals_dict.dur)
    person, speed = infer_duration_lookup(settings, Path(args.result))
    if args.person is not None:
        person = args.person
    if args.speed is not None:
        speed = round(abs(float(args.speed)), 1)

    real_duration = load_real_duration(Path(args.durations_file), person, speed)

    print(f"simulated duration: {sim_duration}")
    if real_duration is not None:
        print(f"experimental duration: {real_duration}")
        print(f"difference: {sim_duration - real_duration}")
    else:
        print("experimental duration: unavailable")
    if person is not None:
        print(f"person: {person}")
    if speed is not None:
        print(f"speed: {speed}")
    print('delta t: ', globals_dict.dur/100)
    print('delay: ', (0.03173078412471625 / (globals_dict.dur/100) ))


if __name__ == "__main__":
    main()
