# Spectra and position records

[Main entry](../README.md)

Optical spectroscopy is supplied as measured data. Differential reflectance is one supported input context. This software does not operate a spectrometer or certify that a requested spectrum was collected.

Install `requirements-analysis.txt`, then run the numeric input/QC path without external models:

```sh
python flakepipeline/spectra_ingest.py --dir /path/to/spectra --substrate "276 nm SiO2/Si" --work /path/to/output --qc-only --no-ai
```

The reader recognizes the existing laboratory spreadsheet/CSV formats. Inspect the retained file identity, energy axis, substrate, fit parameters and flags. A/B exciton fitting and any calibrated stacking rule require the applicable reference set; see `python flakepipeline/spectra_ingest.py --help`. The quoted oxide thickness is an example identifier, not a calibration result for a new substrate.

For classification, provide a substrate-matched calibration file with `--calibrated`. Keep calibration data separate from independent evaluation data. Weak signals, wrong position, mixed coverage and reference acquisition problems require review; morphology alone does not identify hexagonal (2H) or rhombohedral (3R) stacking. A calibrated rule's accuracy is not an accuracy gain caused by a language model.

Retain the microscope point image, actual acquisition footprint, raw sample and reference signals, instrument settings and timing alongside the spectra. Missing position or reference evidence remains missing. The workbench opens available correspondence records and results; it does not generate replacement evidence.
