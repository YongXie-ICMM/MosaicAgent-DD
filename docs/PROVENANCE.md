# Source provenance and scope

This public snapshot starts a new Git history for the Digital Discovery software. The original development repositories, raw experiments, student returns and manuscript histories remain separate and unchanged.

- Mosaic construction and analysis source: author-controlled `MosaicAgent`, commit `43376c1afd45a0e0ad040c342474814c4d64f11f`.
- Acquisition source: author-controlled `AmScope-Camera`, commit `25a839261ae033f068bdcc84507a8cb0e749f760`. [Per-file hashes](../acquisition/acquisition_source.json) and the bundled delivery manifest preserve the source identity.
- Public adaptations: layer-only workbench; explicit local configuration; pure focus-metric module without GUI/driver imports; explicit credential discovery; removal of personal scratch paths; synthetic test fixtures; portable student launchers and documentation.

No earlier Git commits, private manuscripts, raw laboratory journals, proprietary SDK libraries, credentials, or future domain/instance tools are exported into this Git history. A separate student data bundle can include research-authorized images and weights as described in the data guide.

The acquisition runtime is unchanged from the recorded revision. Public bundling and a passing mock test suite do not certify hardware calibration or long-run operation. Source-code publication also does not prove a method's scientific validity.

Python packages and optional vendor camera/stage SDKs retain their respective licenses. Vendor SDK components must be obtained from their suppliers. No new repository-wide redistribution license has been assigned in this initial release. Confirm licensing terms before third-party redistribution or reuse beyond applicable permissions.
