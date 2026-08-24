"""Create BioSym tracking references from Winter (1987), Tables 3.32(a-c).

The appendix reports hip, knee, and ankle mean and standard deviation every
2% of stride for slow, natural, and fast cadence.  Values below are retained
in degrees exactly as tabulated.  Output means are radians and output
variances are rad^2.  BioSym's knee coordinate has the opposite sign from
Winter's flexion-positive convention.
"""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "opensim_exports" / "winter_1987"


SLOW = """
0 15.73 7.01 3.74 5.06 -0.57 3.37
2 14.69 6.69 5.96 5.20 -2.83 3.67
4 13.52 6.43 8.33 5.44 -5.27 3.95
6 12.49 6.35 10.88 5.79 -6.46 3.96
8 11.60 6.45 13.39 6.16 -5.91 3.93
10 10.55 6.64 15.31 6.48 -4.18 3.77
12 9.14 6.84 16.20 6.73 -2.19 3.60
14 7.54 6.99 16.20 6.87 -0.46 3.46
16 6.06 7.01 15.75 6.81 0.96 3.33
18 4.77 6.91 15.07 6.54 2.13 3.21
20 3.50 6.76 14.16 6.14 3.07 3.01
22 2.19 6.64 13.10 5.73 3.79 2.80
24 0.94 6.55 12.04 5.34 4.35 2.66
26 -0.16 6.49 11.10 4.99 4.84 2.62
28 -1.20 6.49 10.28 4.73 5.34 2.66
30 -2.27 6.54 9.54 4.60 5.84 2.76
32 -3.34 6.64 8.93 4.61 6.29 2.89
34 -4.35 6.80 8.47 4.70 6.71 3.01
36 -5.24 7.00 8.23 4.84 7.16 3.11
38 -6.07 7.18 8.21 4.97 7.66 3.27
40 -6.89 7.26 8.36 5.02 8.11 3.47
42 -7.69 7.32 8.71 5.01 8.43 3.57
44 -8.43 7.43 9.33 5.04 8.56 3.60
46 -9.12 7.55 10.25 5.09 8.46 3.60
48 -9.77 7.63 11.53 5.08 8.03 3.63
50 -10.39 7.69 13.21 5.03 7.14 3.76
52 -10.93 7.75 15.47 5.02 5.55 4.01
54 -11.27 7.76 18.54 5.17 3.06 4.49
56 -11.05 7.66 22.74 5.51 -0.34 5.04
58 -9.95 7.41 28.32 5.95 -4.47 5.41
60 -7.96 7.11 35.05 6.36 -8.91 5.29
62 -5.27 6.84 42.28 6.49 -13.12 4.64
64 -2.23 6.66 49.12 6.33 -16.27 4.22
66 0.86 6.44 54.89 5.94 -17.55 4.74
68 3.80 6.10 59.14 5.55 -16.64 5.55
70 6.45 5.70 61.71 5.60 -14.00 5.78
72 8.84 5.44 62.55 6.14 -10.48 5.61
74 11.04 5.31 61.77 6.95 -6.91 5.47
76 13.06 5.25 59.52 7.81 -3.81 5.36
78 14.80 5.29 55.96 8.62 -1.34 5.08
80 16.20 5.33 51.26 9.21 0.51 4.64
82 17.27 5.24 45.57 9.54 1.59 4.28
84 18.07 5.15 39.10 9.63 1.72 4.03
86 18.53 5.21 31.99 9.46 1.11 3.91
88 18.55 5.38 24.44 8.94 0.25 3.93
90 18.17 5.54 16.91 8.02 -0.34 4.03
92 17.55 5.68 10.13 6.68 -0.45 4.09
94 16.88 5.90 4.98 5.11 -0.19 4.01
96 16.30 6.23 2.11 3.72 0.14 3.77
98 15.90 6.51 1.73 3.20 0.04 3.38
100 15.53 6.70 3.21 3.44 -0.89 3.13
"""


FAST = """
0 18.06 5.65 5.91 4.09 1.56 2.71
2 17.58 5.75 9.63 3.83 -0.48 2.76
4 17.06 5.89 13.42 4.26 -2.24 2.86
6 16.62 6.03 17.30 4.60 -2.94 3.08
8 16.21 6.12 20.82 4.76 -2.32 3.45
10 15.62 6.18 23.51 4.83 -0.74 3.73
12 14.58 6.18 25.00 4.87 1.28 3.77
14 13.02 6.09 25.24 4.94 3.25 3.65
16 11.05 5.94 24.48 5.01 4.92 3.49
18 8.87 5.81 22.99 5.06 6.13 3.35
20 6.66 5.73 21.07 5.08 6.92 3.24
22 4.51 5.71 18.93 5.06 7.35 3.16
24 2.47 5.73 16.72 5.00 7.47 3.14
26 0.55 5.79 14.56 4.89 7.36 3.18
28 -1.25 5.91 12.56 4.73 7.16 3.26
30 -2.94 6.10 10.77 4.51 7.02 3.38
32 -4.56 6.30 9.23 4.26 7.04 3.53
34 -6.12 6.49 7.95 4.02 7.22 3.68
36 -7.61 6.67 6.98 3.85 7.49 3.80
38 -8.99 6.82 6.36 3.78 7.74 3.86
40 -10.20 6.94 6.18 3.82 7.91 3.91
42 -11.22 7.04 6.56 3.95 7.95 4.00
44 -12.04 7.14 7.58 4.15 7.74 4.19
46 -12.66 7.23 9.33 4.37 7.11 4.45
48 -13.05 7.32 11.88 4.58 5.80 4.76
50 -13.18 7.42 15.31 4.78 3.53 5.17
52 -12.94 7.56 19.73 5.01 0.07 5.69
54 -12.24 7.70 25.14 5.26 -4.43 6.15
56 -10.95 7.79 31.48 5.51 -9.48 6.25
58 -9.05 7.82 38.48 5.73 -14.28 5.83
60 -6.58 7.76 45.63 5.84 -17.99 5.21
62 -3.62 7.63 52.37 5.75 -19.94 5.11
64 -0.30 7.46 58.15 5.42 -19.95 5.73
66 3.20 7.23 62.58 4.90 -18.28 6.45
68 6.67 6.92 65.39 4.36 -15.49 6.74
70 9.89 6.58 66.52 4.08 -12.26 6.59
72 12.65 6.27 66.05 4.25 -9.09 6.16
74 14.87 5.96 64.09 4.75 -6.25 5.58
76 16.60 5.66 60.84 5.40 -3.84 4.96
78 17.92 5.37 56.46 6.07 -1.92 4.36
80 18.95 5.13 51.08 6.71 -0.52 3.85
82 19.69 5.01 44.79 7.30 0.38 3.49
84 20.06 5.01 37.64 7.76 0.82 3.31
86 19.99 5.12 29.81 7.91 0.97 3.28
88 19.49 5.30 21.74 7.61 1.11 3.32
90 18.70 5.49 14.14 6.79 1.44 3.41
92 17.86 5.63 7.86 5.54 1.95 3.46
94 17.18 5.73 3.65 4.16 2.43 3.42
96 16.76 5.80 2.02 3.25 2.52 3.27
98 16.55 5.83 2.80 3.34 1.93 3.13
100 16.46 5.84 5.47 4.04 0.56 3.08
"""


def parse_table(text: str) -> np.ndarray:
    table = np.loadtxt(text.strip().splitlines())
    if table.shape != (51, 7) or not np.array_equal(table[:, 0], np.arange(0, 101, 2)):
        raise ValueError(f"Unexpected appendix table shape/content: {table.shape}")
    return table


def natural_from_existing_reference() -> np.ndarray:
    """Recover the tabulated natural values from the verified legacy file."""
    source = ROOT / "data" / "opensim_exports" / "kinematics_mean_var.mot"
    lines = source.read_text(encoding="utf-8").splitlines()
    header = lines.index("endheader")
    names = lines[header + 1].split("\t")
    values = np.loadtxt(lines[header + 2 :], delimiter="\t")
    index = {name: idx for idx, name in enumerate(names)}
    rows = []
    for phase in range(0, 100, 2):
        row = values[phase]
        rows.append([
            phase,
            row[index["hip_flexion_l_mean"]],
            np.sqrt(row[index["hip_flexion_l_var"]]),
            -row[index["knee_angle_r_mean"]],
            np.sqrt(row[index["knee_angle_r_var"]]),
            row[index["ankle_angle_l_mean"]],
            np.sqrt(row[index["ankle_angle_l_var"]]),
        ])
    # The 100% row is printed in Table 3.32(b) but omitted from the legacy
    # periodic 100-node file.
    rows.append([100, 19.01, 5.43, 2.21, 3.60, 0.58, 3.52])
    return np.asarray(rows)


def write_raw_csv(label: str, table: np.ndarray) -> None:
    path = OUTPUT / f"winter_1987_{label}_cadence_table_3_32.csv"
    header = "percent_stride,hip_mean_deg,hip_sd_deg,knee_mean_deg,knee_sd_deg,ankle_mean_deg,ankle_sd_deg"
    np.savetxt(path, table, delimiter=",", header=header, comments="", fmt=["%d", *["%.2f"] * 6])


def write_tracking_mot(label: str, table: np.ndarray) -> None:
    source_phase = table[:, 0] / 100.0
    target_phase = np.arange(100) / 100.0
    interpolated = np.column_stack([
        np.interp(target_phase, source_phase, table[:, column])
        for column in range(1, 7)
    ])
    hip_mean, hip_sd, knee_mean, knee_sd, ankle_mean, ankle_sd = interpolated.T

    # Convert Winter's degrees to BioSym radians; knee flexion is negative in
    # the model. The left leg is the right leg shifted by half a gait cycle.
    right_mean = {
        "hip_flexion": np.deg2rad(hip_mean),
        "knee_angle": -np.deg2rad(knee_mean),
        "ankle_angle": np.deg2rad(ankle_mean),
    }
    right_var = {
        "hip_flexion": np.deg2rad(hip_sd) ** 2,
        "knee_angle": np.deg2rad(knee_sd) ** 2,
        "ankle_angle": np.deg2rad(ankle_sd) ** 2,
    }

    columns = ["time"]
    arrays = [target_phase]
    zeros = np.zeros(100)
    for pelvis in ("q_pelvis_tx", "q_pelvis_ty", "q_pelvis_tilt"):
        columns.extend((f"{pelvis}_mean", f"{pelvis}_var"))
        arrays.extend((zeros, zeros))
    for side, shift in (("r", 0), ("l", 50)):
        for joint in ("hip_flexion", "knee_angle", "ankle_angle"):
            columns.extend((f"{joint}_{side}_mean", f"{joint}_{side}_var"))
            arrays.extend((np.roll(right_mean[joint], shift), np.roll(right_var[joint], shift)))

    data = np.column_stack(arrays)
    path = OUTPUT / f"winter_1987_{label}_cadence_kinematics_mean_var_rad.mot"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"name {path.stem}\n")
        stream.write(f"datacolumns {data.shape[1]}\n")
        stream.write(f"datarows {data.shape[0]}\n")
        stream.write("range 0.00000000 0.99000000\nendheader\n")
        stream.write("\t".join(columns) + "\n")
        np.savetxt(stream, data, delimiter="\t", fmt="%.10f")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "slow": parse_table(SLOW),
        "natural": natural_from_existing_reference(),
        "fast": parse_table(FAST),
    }
    for label, table in tables.items():
        write_raw_csv(label, table)
        write_tracking_mot(label, table)
        print(f"Created Winter 1987 {label} cadence references in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
