# Ideal Treadmill BioSym

This repository contains the BioSym optimal-control simulations used to compare
ideal (fixed-speed) and realistic (controller-driven) treadmill walking. The
final analysis uses participants P1-P15, participant-specific experimental GRFs,
Winter (1987) joint-angle references, and a participant-specific planar model
with six Hunt-Crossley contact spheres.

## Local setup

The OpenSim installation used by this project may require its Ipopt shared
library to be preloaded. Set its location for the current shell before using
the Bash launchers or the VS Code launch configuration:

```bash
export IPOPT_LIBRARY="/path/to/opensim-core/sdk/lib/libipopt.so"
```

The path is machine-specific and must not be committed. Copy `.env.example`
to a local `.env` if desired, update the path, and load it before running a
command:

```bash
set -a
source .env
set +a
```

The Bash launchers prepend `IPOPT_LIBRARY` to an existing `LD_PRELOAD`. If the
installed OpenSim/Ipopt combination does not require preloading, the variable
can be left unset.

## Final simulation workflow

For the paper, we first ran the ideal treadmill with the six-sphere contact
stiffness set to `5e5` in the participant's
`gait2d_ground_contact_huntcrossley_6spheres.xml`:

```bash
uv run python pipelines/run_tracking_pipeline_huntcrossley_treadmill.py \
  --participants 3 \
  --speeds 1.0 1.2 1.4 1.6 1.8 \
  --ideal-only \
  --effort-model muscle_volume \
  --muscle-effort-weight 300 \
  --muscle-effort-speedweighting \
  --grf-weight 5e3 \
  --result-label 0814_lowE \
  --no-node-directory \
  --overwrite
```

We then increased the stiffness in the contact XML to `2e6` and reran both
conditions, using the low-stiffness ideal results as initial guesses.

Ideal treadmill:

```bash
uv run python pipelines/run_tracking_pipeline_huntcrossley_treadmill.py \
  --participants 3 \
  --speeds 1.0 1.2 1.4 \
  --ideal-only \
  --initial-guess-label 0814_lowE \
  --effort-model muscle_volume \
  --muscle-effort-weight 300 \
  --muscle-effort-speedweighting \
  --grf-weight 5e3 \
  --result-label 0814_e300_grf5e3_middleStiff \
  --no-node-directory \
  --overwrite
```

Realistic treadmill:

```bash
uv run python pipelines/run_tracking_pipeline_huntcrossley_treadmill.py \
  --participants 3 \
  --speeds 1.0 1.2 1.4 \
  --real-only \
  --initial-guess-label 0814_lowE \
  --effort-model muscle_volume \
  --muscle-effort-weight 300 \
  --muscle-effort-speedweighting \
  --grf-weight 5e3 \
  --result-label 0814_e300_grf5e3_middleStiff \
  --no-node-directory \
  --overwrite
```

## Further documentation

### 1. Create the standing initial guess

Each participant needs a standing solution before the first treadmill solve:

```bash
bash_scripts/run_standing_huntcrossley.sh \
  --config standing_files/standing_p3_6spheres.yaml
```

For P3 this writes:

```text
result_HC/p3/huntcrossley_6spheres/standing.pkl
```

Use the corresponding `standing_pN_6spheres.yaml` for another participant.

For P2-P15, the corresponding participant notebook in `standing_files/` can be
used to compare standing initial guesses and select the best solution.

### 2. Run the ideal treadmill separately

The ideal condition fixes both belt speeds by setting the treadmill-controller
gains to zero internally. By default it starts from the participant's standing
solution; `--initial-guess-file PATH` can select another initial guess.

```bash
uv run python pipelines/run_tracking_pipeline_huntcrossley_treadmill.py \
  --participants 3 \
  --speeds 1.0 1.2 1.4 \
  --ideal-only \
  --effort-model muscle_volume \
  --muscle-effort-weight 300 \
  --muscle-effort-speedweighting \
  --grf-weight 5e3 \
  --result-label 0814_lowE \
  --no-node-directory \
  --overwrite
```

Results are written as, for example:

```text
result_HC/p3/huntcrossley_6spheres/0814_lowE/treadmill_fixed_1_0.pkl
```

The label must identify the ideal run that will later initialize the realistic
run. Objective weights should match the intended simulation condition.

### 3. Run the realistic treadmill separately

The realistic condition uses an existing ideal solution as its warm start and
as the source for detecting controller contact-event boundaries. This is the
final real-only command used for the example P3 results:

```bash
uv run python pipelines/run_tracking_pipeline_huntcrossley_treadmill.py \
  --participants 3 \
  --speeds 1.0 1.2 1.4 \
  --real-only \
  --initial-guess-label 0814_lowE \
  --effort-model muscle_volume \
  --muscle-effort-weight 300 \
  --muscle-effort-speedweighting \
  --grf-weight 5e3 \
  --result-label 0814_e300_grf5e3_middleStiff \
  --no-node-directory \
  --overwrite
```

For each speed, `--initial-guess-label` selects
`treadmill_fixed_<speed>.pkl` from the named ideal-result directory. The
realistic result is saved as `treadmill_real_<speed>.pkl` under the new result
label. A summary PNG and contact-diagnostics CSV are saved alongside each PKL.

`--overwrite` is only necessary when replacing an existing result. Without
`--no-node-directory`, the pipeline inserts a `<N>nodes` directory into both
the result and initial-guess paths.

### Combined and separate pipeline modes

If neither `--ideal-only` nor `--real-only` is supplied, the pipeline runs both
conditions in sequence for each participant and speed:

```text
standing or selected initial guess
  -> ideal treadmill (`treadmill_fixed_<speed>.pkl`)
    -> realistic treadmill (`treadmill_real_<speed>.pkl`)
```

The newly solved ideal result is automatically used as the realistic run's
initial guess. Its contact-diagnostics CSV is also used to define the
realistic controller's contact-event boundaries. No initial-guess label is
needed in this combined mode.

Use `--ideal-only` to stop after the ideal result. Use `--real-only` to skip the
ideal solve; because the automatic ideal result is then unavailable,
`--real-only` requires either `--initial-guess-label LABEL` or
`--initial-guess-file PATH` to select an existing ideal solution.

### Automatic reference selection

The pipeline always selects the participant-specific six-sphere model and GRF
file. It uses Winter natural-cadence angles through 1.4 m/s and Winter
fast-cadence angles at higher speeds.


### Pipeline options

Case and output selection:

- `--participants 1 2 ...`: participant numbers without the `P` prefix.
- `--speeds S1 S2 ...`: walking speeds in m/s.
- `--n-nodes N`: collocation node count (default 100).
- With neither `--ideal-only` nor `--real-only`, run ideal first and
  automatically use that result to initialize realistic.
- `--ideal-only`: run only the fixed-speed treadmill condition.
- `--real-only`: run only the controller-driven condition; requires an existing
  ideal result selected by initial-guess label or file.
- `--initial-guess-label LABEL`: load the corresponding
  `treadmill_fixed_<speed>.pkl` for every requested speed.
- `--initial-guess-file PATH`: use one explicit PKL as the warm start.
- `--result-label LABEL`: name the output subdirectory.
- `--no-node-directory`: omit the `<N>nodes` directory level.
- `--overwrite`: replace results that already exist.
- `--skip-existing`: leave existing results unchanged and continue.
- `--dry-run`: print resolved commands and paths without solving.

Objective settings:

- `--effort-model activation|muscle_volume`: use equal activation weights or
  normalized muscle-volume weights.
- `--muscle-effort-weight VALUE`: override the effort-objective weight.
- `--muscle-effort-speedweighting` / `--no-muscle-effort-speedweighting`:
  enable or disable effort normalization by walking speed raised to the effort
  exponent.
- `--grf-weight VALUE`: override the GRF-tracking weight.
- `--grf-force-threshold VALUE`: track a GRF component only where its measured
  absolute force exceeds this threshold in newtons.
- `--tracking-config PATH`: use another tracking YAML instead of
  `walking_tracking_template.yaml`.

Solver and controller diagnostics:

- `--no-post-hs-interpolation`: use raw anterior-posterior contact force instead
  of post-heel-strike interpolation in realistic runs.
- `--validate-only`: evaluate the initial OCP without running IPOPT.
- `--live-dashboard`: show the solver dashboard.
- `--dashboard-port PORT`: select its port (default 8050).
- `--iteration-log-interval N`: update it every N IPOPT iterations.
- `--visualize`: display the walking visualization after solving.
- `--transform-overground-frame`: transform an overground warm start for the
  combined overground-to-treadmill workflow.

## Participant model preparation

The participant models are already prepared and committed under
`models/gait2d_scaled_p1` through `models/gait2d_scaled_p15`. Model scaling was
a one-time preparation step performed with `scripts/scale_model.py`. That
one-off script was removed during repository cleanup; its generated
participant-specific models are retained as pipeline inputs.

Each retained participant directory therefore contains:

- `gait2d_scaled.xml`: scaled bodies, geometry, masses, and inertias;
- `gait2d_actuators_scaled.xml`: actuator and muscle definitions;
- `gait2d_ground_contact_huntcrossley_6spheres.xml`: contact locations and
  Hunt-Crossley parameters;
- `gait2d_scaled_huntcrossley_6spheres.yaml`: BioSym model configuration linking
  the three XML inputs.

## Experimental-data preparation

### GRF and gait-cycle data

The processed experimental input is `data/gait_data.csv`. Each row belongs to
one participant, walking speed, and normalized gait-cycle node. The relevant
columns are:

- `Fx1`, `Fy1`: right anterior-posterior and vertical GRF;
- `Fx2`, `Fy2`: left anterior-posterior and vertical GRF;
- `speed1`, `speed2`: right and left belt speed;
- `duration`: experimental gait-cycle duration.

The participant- and speed-specific OpenSim MOT files were created with
`export_opensim_mot.py`. This was a one-time data-preparation script and was
removed during repository cleanup.

The resulting files are committed in `data/opensim_exports`. `src/gait_data.py`
loads the CSV, validates its columns, selects participant-speed cases, and
periodically resamples GRFs, belt speeds, or duration-dependent data when a
different node count is requested.

The raw, uncprocessed gait data can be found in XX (LINK REPO)

### Winter joint-angle references

The retained joint-angle targets come from Winter (1987), Tables 3.32(a-c).
Regenerate them with:

```bash
uv run python scripts/create_winter_1987_cadence_references.py
```

The raw transcriptions and generated MOT files are under
`data/opensim_exports/winter_1987`; their local README documents the source
tables and sample sizes.

## How the pipeline, Bash launcher, and tracking script differ

The three layers have different responsibilities:

```text
pipeline (many requested cases and file selection)
  -> Bash launcher (runtime environment)
    -> tracking script (one BioSym optimization)
```

### Pipeline scripts

The files under `pipelines/` are the user-facing batch runners. They loop over
participants and speeds, select the correct participant model, GRF file, Winter
angle file, initial guess, and output path, and decide whether to run the ideal
or realistic condition. A pipeline does not construct the BioSym OCP itself;
for every case it assembles a command and calls the appropriate Bash launcher.

| Pipeline | What it runs | Initial-guess flow | Intended use |
| --- | --- | --- | --- |
| `run_tracking_pipeline_huntcrossley_treadmill.py` | Ideal and/or realistic treadmill | By default, standing → ideal → realistic. With an `--only` flag, run one selected branch. | Final treadmill pipeline used for the paper. |
| `run_tracking_pipeline_huntcrossley_overground.py` | Overground walking only | Standing or `--initial-guess-file` → overground. | Generate an overground solution with the same participant-specific six-sphere model and tracking data. |
| `run_overground_to_ideal_realistic_treadmill.py` | Overground, ideal treadmill, and realistic treadmill | Overground is solved first; ideal and realistic then both start independently from that same overground result. Realistic does not start from ideal in this pipeline. | Compare both treadmill branches against a common overground-derived seed. |

`pipelines/run_tracking_pipeline_huntcrossley_treadmill.py` is therefore the
pipeline to use for the final paper workflow described at the beginning of this
README. Its default behavior automatically chains ideal into realistic.

`pipelines/run_tracking_pipeline_huntcrossley_overground.py` uses
`walking_tracking_overground.yaml` and the stationary-ground tracking script.
It produces `overground_<speed>.pkl` and does not run either treadmill
condition.

`pipelines/run_overground_to_ideal_realistic_treadmill.py` is deliberately a
different experiment from the default treadmill chain. After solving
overground, it creates two parallel branches:

```text
overground result
  +-> ideal treadmill
  +-> realistic treadmill
```

With `--transform-overground-frame`, it first removes forward pelvis
progression and initializes both belts at the target speed. The transformed
result is then the common warm start for both branches.

`pipelines/tracking_pipeline_common.py` is not a runnable pipeline. It contains
the common argument parsing, Winter and GRF selection, output naming, command
construction, and subprocess handling imported by the three pipeline entry
points.

The pipeline commands shown above are the normal way to generate multiple
paper results.

### Bash launchers

The files under `bash_scripts/` are environment wrappers, not batch pipelines.
They do not select participants, speeds, models, or result labels. They set the
repository root and Python path, configure a writable Matplotlib cache, preload
the required IPOPT library, apply the local BioSym Python 3.11 compatibility
patch, and then forward all arguments to one Python solver script.

- `bash_scripts/run_tracking.sh` launches one treadmill solve through
  `script_tracking.py`.
- `bash_scripts/run_tracking_overground.sh` launches one overground solve
  through `script_tracking_overground.py`.
- `bash_scripts/run_standing_huntcrossley.sh` launches one standing solve
  through `scripts/run_standing_huntcrossley.py`.

The pipelines call these launchers internally. Run a Bash launcher directly
only when debugging or manually running one fully specified case.

### Python solver scripts

- `script_tracking.py` loads the model and tracking YAML, installs the custom
  objectives and treadmill constraint, constructs one treadmill BioSym OCP,
  solves it, and saves its result and diagnostics.
- `script_tracking_overground.py` performs the equivalent work for one
  overground OCP.
- `scripts/run_standing_huntcrossley.py` constructs and solves one
  participant-specific standing equilibrium problem.

### Configuration and source modules

- `walking_tracking_template.yaml`: node count, bounds, objectives, constraints,
  and default weights for treadmill tracking.
- `ocp_controller_params.json`: realistic treadmill controller gains and timing
  indexed by node count.
- `src/contact_huntcrossley.py`: repository-local OpenSim-compatible
  Hunt-Crossley force implementation registered with BioSym.
- `src/gc_model_huntcrossley_treadmill.py`: adds the two belt-speed states to the
  Hunt-Crossley ground-contact model.
- `src/treadmill_constraint_swing_target.py`: OCP constraint for the realistic
  treadmill controller, including swing-phase target-speed behavior.
- `src/treadmill_controller.py`: shared controller equations, filtering, and
  post-heel-strike AP-force interpolation.
- `src/track_grf_contact.py`: GRF tracking objective used with the contact model.
- `src/track_duration.py`: duration tracking objective using
  `data/gait_data.csv`.
- `src/muscle_volume_weighted_effort.py`: muscle-volume effort objective and
  optional speed weighting.
- `src/pooled_angle_variance.py` and
  `src/track_angles_pooled_variance.py`: current across-cadence variance pooling
  for angle tracking.
- `visualize_walking2d.py`: loads saved results and creates diagnostic summary
  plots.
- `scripts/export_simulation_results_csv.py`: exports completed simulations to
  analysis-ready CSV files, including normalized GRFs and metabolic variables.

## Output layout

With `--no-node-directory`, results follow:

```text
result_HC/
  pN/
    huntcrossley_6spheres/
      standing.pkl
      <ideal-label>/
        treadmill_fixed_<speed>.pkl
        treadmill_fixed_<speed>.png
        treadmill_fixed_<speed>_contact_diagnostics.csv
      <realistic-label>/
        treadmill_real_<speed>.pkl
        treadmill_real_<speed>.png
        treadmill_real_<speed>_contact_diagnostics.csv
```

Use `--dry-run` on a pipeline command to print all resolved input and output
paths without starting an optimization.
