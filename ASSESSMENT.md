# Assessment and salvage decision — 2026-10-14

## What was actually present

GenTraj4Embodied is an evolved version of the earlier local trajectory-generation prototype. Of 15 core Python modules, 13 were byte-identical; all four experiment scripts were identical. The model registry and runner were improved, and this copy contained substantially more MLP training evidence.

Local logs recorded an MLP run through epoch 99, ending at training loss 0.020639. A retained checkpoint from another run, epoch 27, reports training loss 0.02194426542320665 and `val_loss=None`. Training loss is not held-out task success.

A saved visualization evaluation contains only five samples, with mean endpoint position error 0.9604926705360413 in its reported coordinate units. Its visualization pipeline can skip trajectories it considers unreasonable, so that file is not an unbiased full-test-set success estimate. This is an observed historical artifact, not a new evaluation reproduced here.

The local manuscript mixes preliminary MLP results with unsupported multi-model success tables. Those claims have not been validated. The manuscript, checkpoints and local logs were not previously tracked in the public repository and are not introduced as verified public research results by this update.

## Decision

Do not spend a low-cost rescue budget on a CCF-B/C submission or a second competing model zoo. Neither additional model names nor manuscript polish establishes a scientific contribution. The practical salvage is a standalone data inspector that can be used before training any trajectory model.

The maintained product now comprises a minimal audit CLI, adversarial unit tests, a standalone interactive report, and reproducible evidence from the repository's existing validation/test files. The older implementation remains reference material, not an endorsed training pipeline.

## New, reproducible evidence

`docs/report.json` covers 2,027 validation and 2,028 test trajectories. All 4,055 records pass the explicitly implemented integrity checks; lengths vary from 45 to 54. There are 4,055 distinct rounded endpoint conditions and no cross-file exact trajectory duplicates. No training-set claim is made: the training file had already been deleted from the local working copy before this task (it remains in the existing Git history).

This reveals an evaluation-design limitation: there are no repeated endpoint reference groups in these two files. It does not prove a lack of latent multimodal structure or an inability to learn a conditional distribution.

The new inspector computes position derivatives with real time intervals and canonicalizes quaternion sign for endpoint grouping. Source fingerprints and per-record findings allow others to reproduce and inspect the audit. Its scope explicitly excludes collision/IK feasibility, robot task success, and model distribution quality.

## Acceptance

- Audit valid, malformed and missing files without importing legacy model modules.
- Detect endpoint inconsistency, nonunit quaternions, nonmonotone time, and cross-file exact duplicates.
- Verify nonuniform-time derivative calculations and quaternion sign equivalence.
- Rebuild the published JSON from the checked-in source HDF5 files in CI.
- Provide a readable interactive report and concise installation/schema documentation.

No acceptance probability, novelty claim, or paper submission is implied by this salvage decision.


## Report revision — 2026-10-14

Added a sourced Chinese research assessment, evidence boundaries, estimated next-step costs and explicit stopping criteria. Expanded the live report with quantitative distribution diagnostics and research context from Diffusion Policy, π₀/π₀.₅ and LeRobot. No new model-training claims. Local numerical checks and mobile layout checks passed; GitHub Actions independently validates the published revision. See RESEARCH_REPORT.zh-CN.md for the detailed decision.

After remote publication and verification, local working copies and task-owned temporary materials are scheduled for deletion at the owner’s request. Public repositories and the existing private preservation archive remain the recovery sources.
