# Winter 1987 cadence-specific joint-angle references

These files transcribe Tables 3.32(a-c), printed on pages 57-59 of David A.
Winter, *The Biomechanics and Motor Control of Human Gait* (1987). The source
PDF is `data/Winter_1987_TheBiomechanicsandMotorControlofHumanGait-2.pdf`
(PDF pages 65-67).

- `*_table_3_32.csv` retains the appendix values at 2% stride intervals in
  degrees and standard deviation.
- `*_kinematics_mean_var_rad.mot` interpolates them to the project's
  100-node periodic convention. Means are radians and variances are rad².
- Winter reports one limb. The MOT files create the left limb by shifting the
  right limb by 50% of the gait cycle.
- Winter uses positive knee flexion; the MOT files negate knee means to match
  BioSym's coordinate convention. Hip and ankle signs are unchanged.
- Sample sizes are slow N=19, natural N=19, and fast N=17.

Regenerate with:

```bash
uv run python scripts/create_winter_1987_cadence_references.py
```
