# Implemented architecture

[Main entry](../README.md) · [中文入口](../README_zh.md)

| Responsibility | Source | Boundary |
|---|---|---|
| Instrument acquisition | `acquisition/Auto_Scan/` | Windows camera/stage GUI; hardware actions remain explicit. |
| Acquisition-to-stitching contract | `stitch_profile.py`, `tools/stitch_setup/` | Checks image geometry and available session metadata; a profile is bound to inspected inputs. |
| Tile processing and mosaics | `run_stitch.py`, `tiles.py`, `register.py`, `blend.py` | Numeric quality metrics, registration, support/coverage checks and compositing. |
| Model roles and calls | `kimi_agents.py`, `flakepipeline/region_agents.py` | Optional image interpretation and region review; outputs are recorded. |
| Stage controller | `flakepipeline/orchestrator.py` | Fixed stage graph and bounded named repair policies; not an autonomous planner of arbitrary experiments. |
| Semantic layer inference | `flakepipeline/seg_model.py`, `flakepipeline/stages.py` | U-Net pixel classes, valid support and statistical calculation; no instance splitting. |
| Review and provenance | `flakepipeline/review.py`, stage manifests | Review queues and recorded decisions; keep underlying measurements and products. |
| Spectral processing | `flakepipeline/spectra_ingest.py`, `flakepipeline/autospectra.py` | Ingest, numerical QC, A/B peak fitting and calibrated rules; independent from instrument acquisition. |
| User entry | `tools/analysis_workbench/` | Local launch/review interface; serves configured source products. |

The post-acquisition controller runs stitch → segment → audit → regions → statistics. Each stage returns status, evidence and products. Named failures can trigger a bounded deterministic parameter change when repairs are enabled; unresolved cases enter human review. A region verdict is an operational decision, not an independent physical label. The default example and student layer demo disable model calls and repairs.

The optional model may interpret tiles, describe segmentation discrepancies, or adjudicate proposed exclusion regions. Numeric operators perform registration, mask generation and counting. The image audit reports discrepancies without altering the predicted mask. Some inherited role prompts and console messages remain in Chinese to preserve the recorded behavior; this release does not translate prompts and silently claim equivalence.

Source inputs, model identity, parameter settings, excluded support and denominators must stay traceable. The current pipeline is not a demonstrated general scientific agent, and this code release provides no superiority claim over a generic agent. Replay of recorded decisions does not establish their physical correctness.

Instance annotation, touching-domain partitioning, orientation prediction, autonomous selection of new instruments and active spectroscopy acquisition are not included in this submission snapshot. The spectral tool processes supplied measurements; a request for spectra is not a measurement.
