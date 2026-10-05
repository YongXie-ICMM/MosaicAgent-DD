# Layer and spectral analysis

Use the [repository README](../README.md) for installation and the runnable student demo. The analysis CLI is `python flakepipeline/run.py --config configs/layer_analysis.example.json --no-ai --no-repair`, launched from the repository root after supplying the configured inputs.

This directory contains semantic U-Net inference, image audit, region exclusion review, area statistics and a separate measured-spectrum analysis path. It contains no instance annotation or touching-domain separation tool. `autoscan_agent.py` is a compatibility entry for offline focus metrics; physical acquisition is in `acquisition/Auto_Scan/`.

For a headless numeric focus check: `python -m flakepipeline.focus_metrics --check-focus /path/to/tiles`.
