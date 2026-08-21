from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GAIT_DATA_FILE = Path("data/gait_data.csv")


def participant_label(participant) -> str:
    if isinstance(participant, str):
        value = participant.strip()
        return value if value.upper().startswith("P") else f"P{int(value)}"
    return f"P{int(participant)}"


@lru_cache(maxsize=8)
def load_gait_data(datafile: str | Path = DEFAULT_GAIT_DATA_FILE) -> pd.DataFrame:
    data = pd.read_csv(datafile)
    required_columns = {
        "participant",
        "speed",
        "node",
        "Fx1",
        "Fy1",
        "Fx2",
        "Fy2",
        "speed1",
        "speed2",
        "duration",
    }
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"{datafile} is missing required columns: {sorted(missing)}")
    return data


def available_participants(datafile: str | Path = DEFAULT_GAIT_DATA_FILE) -> list[int]:
    data = load_gait_data(datafile)
    labels = data["participant"].dropna().unique().tolist()
    return sorted(int(str(label).strip().lstrip("Pp")) for label in labels)


def get_gait_case(datafile: str | Path, participant, speed: float) -> pd.DataFrame:
    data = load_gait_data(datafile)
    person = participant_label(participant)
    rounded_speed = round(abs(float(speed)), 1)
    case = data[
        (data["participant"].astype(str).str.upper() == person.upper())
        & np.isclose(data["speed"].astype(float), rounded_speed)
    ].copy()
    if case.empty:
        raise ValueError(f"No gait data found for participant {person}, speed {rounded_speed:.1f}.")
    return case.sort_values("node").reset_index(drop=True)


def get_duration(datafile: str | Path, participant, speed: float) -> float:
    case = get_gait_case(datafile, participant, speed)
    durations = case["duration"].dropna().astype(float).unique()
    if durations.size == 0:
        raise ValueError(f"No duration found for participant {participant}, speed {speed}.")
    return float(durations[0])


def resample_periodic_series(values, n_nodes: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == int(n_nodes):
        return values.copy()

    old_x = np.arange(values.size + 1, dtype=float) / values.size
    new_x = np.arange(int(n_nodes), dtype=float) / int(n_nodes)
    return np.interp(new_x, old_x, np.concatenate([values, values[:1]]))


def get_belt_speeds(datafile: str | Path, participant, speed: float, n_nodes: int | None = None) -> dict[str, np.ndarray]:
    case = get_gait_case(datafile, participant, speed)
    right = case["speed1"].to_numpy(dtype=float)
    left = case["speed2"].to_numpy(dtype=float)
    if n_nodes is not None:
        right = resample_periodic_series(right, n_nodes)
        left = resample_periodic_series(left, n_nodes)
    return {"right": right, "left": left}


def get_grfs(datafile: str | Path, participant, speed: float, n_nodes: int | None = None) -> dict[str, np.ndarray]:
    case = get_gait_case(datafile, participant, speed)
    grfs = {
        "fx_r": case["Fx1"].to_numpy(dtype=float),
        "fy_r": case["Fy1"].to_numpy(dtype=float),
        "fx_l": case["Fx2"].to_numpy(dtype=float),
        "fy_l": case["Fy2"].to_numpy(dtype=float),
    }
    if n_nodes is not None:
        grfs = {key: resample_periodic_series(value, n_nodes) for key, value in grfs.items()}
    return grfs
