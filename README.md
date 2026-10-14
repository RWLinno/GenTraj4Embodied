# GenTraj4Embodied · Pose Trajectory Inspector

**Check your trajectory data before trusting your model scores.**

[Open the audit report →](https://rwlinno.github.io/GenTraj4Embodied/) · [中文说明](README.zh-CN.md) · [Companion diffusion playground](https://rwlinno.github.io/trajectory-playground/)

A lightweight, model-free audit CLI for HDF5 pose trajectories. It produces a machine-readable JSON report and a standalone interactive HTML report with trajectory projections. Only NumPy and h5py are required; no GPU, PyTorch, model weights, or server is needed to run the audit.

This repository began as a broad trajectory-generation prototype. The maintained entry point is now **`python -m inspector.audit`**. The old model and training files are retained as experimental reference, not as a verified model benchmark. See [the project assessment](ASSESSMENT.md).

## What it checks

- Array dimensions, finite values, and strictly increasing timestamps.
- Unit quaternion integrity and start/end pose agreement. Quaternion sign is treated correctly: `q` and `-q` describe the same orientation.
- Position speed and acceleration computed from **actual, possibly nonuniform timestamps**.
- Record-count and per-record length metadata consistency. Variable-length trajectories are supported.
- Exact stored trajectory duplicates across files, and repeated endpoint conditions using six-decimal pose rounding.
- Input SHA-256 fingerprints, task counts, waypoint-length distributions and per-record findings.

It **does not** certify robot safety, collision freedom, IK feasibility, joint limits, task success, or generative-model quality. Coordinate units are inherited from the input. Repeated endpoint groups ignore scene/task metadata and are informational, not automatic evidence of leakage or valid multimodality.

## Run in two commands

Use Python 3.10+ in a virtual environment:

```bash
python -m pip install -r inspector/requirements.txt
python -m inspector.audit experiments/data/val.h5 experiments/data/test.h5 --out audit-output
```

Open `audit-output/index.html` directly in your browser. It contains the report data and has no external script, font, or network dependency. `report.json` includes every record's statistics. Input files are read-only.

For your own data:

```bash
python -m inspector.audit train.h5 validation.h5 test.h5 --out audit-output --strict
```

`--strict` exits 1 if there are integrity flags, missing/unreadable inputs, or exact cross-file trajectory duplicates. Endpoint overlaps alone are informational. Without it, the tool generates the report even when it finds problems. Reusing an output directory replaces its report files.

### Supported HDF5 layout

Each trajectory is a top-level group, with four required datasets:

```
trajectory_0/
  trajectory    float[N,7]    # x,y,z plus four quaternion components
  timestamps    float[N]     # strictly increasing; units define speed units
  start_pose    float[7]
  end_pose      float[7]
```

Optional group attributes: `task_name`, `modality`, `length`. Optional root attributes: `num_trajectories`, `trajectory_length` (reported as nominal only). Quaternion component order must be consistent across trajectory and endpoints; these norm/sign/dot-product checks do not require choosing xyzw vs wxyz. At least two waypoints are required. Use distinct filenames for the inputs. Other HDF5 layouts need an adapter; this tool does not guess schemas.

## Published example: real files from this prototype

[Full report](docs/report.json) · [Interactive report](https://rwlinno.github.io/GenTraj4Embodied/)

| Check | Observed result |
|---|---:|
| Validation + test trajectories | 4,055 |
| Record integrity flags under the listed checks | 0 |
| Waypoints per trajectory | 45–54 |
| Unique endpoint conditions | 4,055 |
| Repeated endpoint conditions | 0 |
| Exact cross-file trajectory duplicates | 0 |
| Median per-trajectory maximum speed | 0.6145 coordinate units / time unit |
| Median per-trajectory maximum acceleration | 6.1451 coordinate units / time unit² |

A zero-flag report means these specific checks passed. It does not imply task feasibility or model readiness. The published audit covers `val.h5` and `test.h5` only, not the training set. Acceleration here is the difference of interval velocities divided by the distance between interval mid-times; it is a finite-difference diagnostic, not a smoothed physical measurement.

The absence of repeated endpoint groups explains why this dataset cannot directly provide a fixed-condition reference bank for measuring multimodal coverage. It does not mathematically prove that a generative model cannot learn a conditional distribution from such data.

## Verify and extend

```bash
python -m unittest discover -s tests -v
python -m inspector.audit experiments/data/val.h5 experiments/data/test.h5 --out audit-output --strict
python scripts/verify_inspector_report.py audit-output/report.json docs/report.json
```

CI exercises nonuniform timing, quaternion sign equivalence, invalid inputs, endpoint mismatches, duplicates, safe HTML serialization and strict exit behavior, then regenerates the published audit and compares all fields to the committed JSON (metadata/counts exact; absolute floating-point tolerance 1e-7, relative tolerance 1e-10). Trajectory previews show the first eight readable records per file in lexicographic group order; the audit statistics always use all records.

Useful extensions include adapters for other dataset layouts, angular-velocity diagnostics, and optional scene-aware collision checks. Keep those checks explicit rather than labeling a generic score “physical feasibility”.

## Relationship to Trajectory Playground

[Trajectory Playground](https://github.com/RWLinno/trajectory-playground) teaches why squared-error regression can average two valid routes into a collision. This repository addresses the complementary question: **does the data and evaluation protocol support the claim you want to make?**

## License

Apache-2.0, as in the original repository. The new inspector and report template use the same license. Historical prototype code has not been certified by the inspector tests; the unpublished manuscript and old result tables are not evidence of validated comparative performance.


## Research context and diagnostics · 2026-10-14

[中文研究评估与低成本路线](RESEARCH_REPORT.zh-CN.md) covers verified evidence, diffusion / flow-matching / VLA context, practical next steps, and stopping criteria. The live report now includes distribution diagnostics; research references are context, not implemented baselines.
